import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from utils.tools import seed_everything, ClassifierDataset, save_model
from utils.hyperParams import get_parser, print_arguments
from utils.processor import Processor
from models.bert import BertClassifier
from loguru import logger
from evaluate import evaluate


def train():
    logger.add("log_bak/teacher_train.log")

    args = get_parser()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    print_arguments(args, logger)

    processor = Processor(args)

    # load data
    train_loader = DataLoader(
        dataset=ClassifierDataset(processor.convert_examples_to_features('train')),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=ClassifierDataset.collate_fn
    )
    valid_loader = DataLoader(
        dataset=ClassifierDataset(processor.convert_examples_to_features('test')),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=ClassifierDataset.collate_fn
    )

    t_total = len(train_loader) * args.teacher_num_epochs

    # define model
    model = BertClassifier.from_pretrained(args.bert_path, processor.tokenizer).to(device)

    # define the optimizer
    bert_parameters = model.bert.named_parameters()
    other_parameters = model.classifier.named_parameters()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': args.bert_lr},
        {'params': [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.bert_lr},

        {'params': [p for n, p in other_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': args.other_lr},
        {'params': [p for n, p in other_parameters if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.other_lr}
    ]

    warmup_step = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                num_training_steps=t_total)

    logger.info('\n')
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = {}".format(args.teacher_num_epochs))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
        args.batch_size))
    logger.info("  Total optimization steps = {}".format(t_total))
    logger.info("  lr of encoder = {}, lr of task_layer = {}".format(
        args.bert_lr, args.other_lr))
    logger.info('\n')

    # train
    best_result = 0
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.teacher_num_epochs):
        model.train()
        logger.info('\n')
        for step, batch in enumerate(train_loader):
            model.zero_grad()

            input, label, text = batch
            # to device
            for key in input.keys():
                if key not in ['label']:
                    input[key] = input[key].to(device)
            label = label.to(device)

            logits = model(**input)
            loss = loss_fn(logits, label)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                logger.info('epoch: %2d, step: %4d, loss: %6.4f' % (epoch, step, loss.item()))

        valid_loss, valid_acc = evaluate(model, valid_loader, loss_fn, device)
        logger.info('epoch: %2d, valid_loss: %4f, valid_acc: %6.4f' % (epoch, valid_loss, valid_acc * 100))
        if valid_acc > best_result:
            best_result = valid_acc
            save_model(model, logger, './checkpoints/teacher_model.pth')

        logger.info('best_result: %6.4f' % (best_result*100))



if __name__ == '__main__':
    train()
