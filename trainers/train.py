import argparse

import os

import logging

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    BertConfig, EncoderDecoderConfig, EncoderDecoderModel, BertForMaskedLM,
)

from .args import get_args

from .data_processing import dataProcessor, cardDataset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Accuracy metrics.
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Loggers.
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, y_labels):

    output_str = args.output_dir.split("/")[-1]
    comment_str = "_{}".format(output_str)
    tb_writer = SummaryWriter(comment=comment_str)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) \
                                // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps \
                  * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )






    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed "
        "& accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps,
    )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch",
        disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)  # Added here for reproductibility.



    class_0 = np.sum(y_labels == 0)
    class_1 = np.sum(y_labels == 1)

    total = len(y_labels)

    weight_for_0 = (1 / class_0)*(total)/2.0 
    weight_for_1 = (1 / class_1)*(total)/2.0

    class_weights = torch.tensor([weight_for_0, weight_for_1]).to(args.device)

    loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            model.train()

            # Processes a batch.
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "labels": batch[3]}



            outputs = model(**inputs)

            loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), inputs["labels"].view(-1))



            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()


            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if (args.logging_steps > 0
                    and global_step % args.logging_steps == 0):
                    # Log metrics
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer,
                                           data_split=args.eval_split)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                "eval_on_{}_{}".format(args.eval_split, key),
                                value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0],
                                         global_step)
                    tb_writer.add_scalar("loss",
                        (tr_loss - logging_loss) / args.logging_steps,
                        global_step)
                    logging_loss = tr_loss


            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step




def load_and_cache_examples(args, tokenizer, evaluate=False,
                            data_split="test"):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = dataProcessor()

    # Getting the examples.
    if data_split == "test" and evaluate:
        examples = processor.get_test_examples()
    elif (data_split == "val" or data_split == "dev") and evaluate:
        examples = processor.get_dev_examples()
    elif data_split == "train" and evaluate:
        examples = processor.get_train_examples()
    elif "test" == data_split:
        examples = processor.get_test_examples()
    else:
        examples = processor.get_test_examples() if evaluate else processor.get_train_examples()
        

    logging.info("Number of {} examples in task: {}".format(
        data_split, len(examples[0])))

    # Defines the dataset.
    dataset = cardDataset(examples, tokenizer,
                                 max_seq_length=args.max_seq_length)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier() 
    
    return dataset, examples[1]




def evaluate(args, model, tokenizer, prefix="", data_split="test"):

    # Main evaluation loop.
    results = {}
    eval_dataset, _ = load_and_cache_examples(args, 
                                           tokenizer, evaluate=True,
                                           data_split=data_split)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size=args.eval_batch_size)



    # Eval!
    logger.info("***** Running evaluation on split: {} {} *****".format(
        data_split, prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    labels = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            # Processes a batch.
            inputs = {"input_ids": batch[0], "attention_mask": batch[1]}

            inputs["labels"] = batch[3]

            inputs["token_type_ids"] = batch[2] 

            outputs = model(**inputs)

            loss = outputs.loss
            logits = outputs.logits

            eval_loss += loss.mean().item()

            logits = torch.nn.functional.softmax(logits, dim=-1)

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()

            labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

        if args.max_eval_steps > 0 and nb_eval_steps >= args.max_eval_steps:
            logging.info("Early stopping evaluation at step: {}".format(args.max_eval_steps))
            break

    # Organize the predictions.
    preds = np.reshape(preds, (-1, preds.shape[-1]))
    preds = np.argmax(preds, axis=-1)

    # Computes overall average eavl loss.
    eval_loss = eval_loss / nb_eval_steps

    eval_loss_dict = {"loss": eval_loss}
    results.update(eval_loss_dict)

    eval_acc = 0
    eval_prec = 0
    eval_recall = 0
    eval_f1 = 0

    eval_acc = accuracy_score(labels, preds)
    eval_prec = precision_score(labels, preds, average = args.score_average_method)
    eval_recall = recall_score(labels, preds, average = args.score_average_method)
    eval_f1 = f1_score(labels, preds, average = args.score_average_method)

    eval_acc_dict = {"{}_accuracy".format(args.task_name): eval_acc}
    eval_acc_dict["{}_precision".format(args.task_name)] = eval_prec
    eval_acc_dict["{}_recall".format(args.task_name)] = eval_recall
    eval_acc_dict["{}_F1_score".format(args.task_name)] = eval_f1

    results.update(eval_acc_dict)

    output_eval_file = os.path.join(args.output_dir,
        prefix, "eval_results_split_{}.txt".format(data_split))

    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} on split: {} *****".format(prefix, data_split))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
            writer.write("%s = %s\n" % (key, str(results[key])))

    return results




def main():
    torch.autograd.set_detect_anomaly(True)
    
    args = get_args()

    # Writes the prefix to the output dir path.
    if args.output_root is not None:
        args.output_dir = os.path.join(args.output_root, args.output_dir)
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use "
            "--overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    else:
        os.makedirs(args.output_dir, exist_ok=True)



    
    device = torch.device("cuda" if torch.cuda.is_available()
                            and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.device = device

    # Setup logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed "
        "training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Sets seed.
    set_seed(args)



    # Getting the labels
    processor = dataProcessor()
    num_labels = processor.get_labels()




    model_checkpoint = args.model_name_or_path
    config = AutoConfig.from_pretrained(model_checkpoint)


    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=False, config=config)

    
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, config=config)

    # Loads models onto the device (gpu or cpu).
    model.to(args.device)
    print(model)
    args.model_type = config.model_type

    logger.info("Training/evaluation parameters %s", args)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("!!! Number of Params: {} M".format(count_parameters(model)/float(1000000)))

    # Training.
    if args.do_train:
        train_dataset, y_labels = load_and_cache_examples(args, tokenizer, data_split="train",
                                                evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, y_labels)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    # Evaluation.
    # results = {}
    # if args.do_eval and args.local_rank in [-1, 0]:
    #     checkpoints = [args.output_dir]
    #     if args.eval_all_checkpoints:
    #         checkpoints = list(
    #             os.path.dirname(c) for c in sorted(glob.glob(
    #                 args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    #         )
    #     else:
    #         assert args.iters_to_eval is not None, ("At least one"
    #             " of `iter_to_eval` or `eval_all_checkpoints` should be set.")
    #         checkpoints = []
    #         for iter_to_eval in args.iters_to_eval:
    #             checkpoints_curr = list(
    #                 os.path.dirname(c) for c in sorted(glob.glob(
    #                     args.output_dir + "/*-{}/".format(iter_to_eval)
    #                     + WEIGHTS_NAME, recursive=True))
    #             )
    #             checkpoints += checkpoints_curr

    #     logger.info("\n\nEvaluate the following checkpoints: %s", checkpoints)
    #     for checkpoint in checkpoints:
    #         logger.info("\n\nEvaluate checkpoint: %s", checkpoint)
    #         global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    #         prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
    #         ckpt_path = os.path.join(checkpoint, "pytorch_model.bin")
    #         model.load_state_dict(torch.load(ckpt_path))
    #         model.to(args.device)

    #         ##################################################
    #         # TODO: Make sure the eval_split is "test" if in
    #         # testing phase.
    #         pass  # This TODO does not require any actual
    #               # implementations, just a reminder.
    #         # End of TODO.
    #         ##################################################

    #         result = evaluate(args, model, tokenizer, prefix=prefix, data_split=args.eval_split)
    #         result = dict((k + "_{}".format(global_step), v)
    #                        for k, v in result.items())
    #         results.update(result)

    # return results


if __name__ == "__main__":
    main()