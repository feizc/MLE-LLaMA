import argparse 
import torch 
import os 
from tqdm import tqdm 
from torch import optim 
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tensorboard
from torch.cuda.amp import GradScaler

from llama import LlamaTokenizer, LlamaForCausalLM
from utils import world_info_from_env, init_distributed_device, TextDataSet, is_master, get_autocast


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a llama model on a causal language modeling task")
    parser.add_argument(
        "--train_file", type=str, default='./data/alpaca_data.json', help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default='./ckpt',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--output_dir", type=str, default='out', help="Where to store the final model.")
    parser.add_argument(
        "--tensorboard_path", type=str, default="./tensorboard",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--norm_gradient_clip", type=float, default=1.0, help="Gradient clip."
    )
    parser.add_argument("--beta1", type=float, default=0.98, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank.")
    
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bfloat16", "fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    args = parser.parse_args()
    return args 


def main():
    args = parse_args()
    print(args) 
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    
    # discover initial world args early so we can log properly
    args.distributed = False
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # fully initialize distributed device environment
    device = init_distributed_device(args)
    
    if is_master(args):
        if not os.path.exists(args.tensorboard_path): 
            os.makedirs(args.tensorboard_path)
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    else:
        writer = None


    tokenizer = LlamaTokenizer.from_pretrained(args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(args.model_name_or_path) 
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps,)
    scaler = GradScaler() if args.precision == "amp" else None

    train_dataset = TextDataSet(args.train_file, tokenizer=tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size) 

    
    for epoch in range(args.num_train_epochs): 
        model.train()
        device = torch.device(args.device)
        autocast = get_autocast(args.precision) 

        num_batches_per_epoch = len(train_loader)
        loss_cum = .0
        progress = tqdm(total=len(train_loader), desc='llama fine-tuning') 

        for i, batch in enumerate(train_loader):  
            step = num_batches_per_epoch * epoch + i
            input_ids, attention_mask = batch['input_ids'].squeeze(0), batch['attention_mask'].squeeze(0)
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device) 
            optimizer.zero_grad() 

            with autocast(): 
                loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            if scaler is not None:
                scaler.scale(loss).backward()
     
                if args.norm_gradient_clip is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.norm_gradient_clip, norm_type=2.0)
                scaler.step(optimizer)
                scaler.update()
            else: 
                loss.backward() 
                optimizer.step() 
            
            loss_cum += loss.item()
            progress.set_postfix({"loss": loss_cum / (i + 1)})
            progress.update() 
            if is_master(args) and  i % 10 == 0: 
                writer.add_scalar("train/loss", loss.item(), step)
            break 
        break 

    if is_master(args):
        print('save modeling')
        torch.save(model.state_dict(), args.output_dir + str(epoch) + '.pt') 
        torch.cuda.synchronize()




if __name__ == "__main__":
    main()






