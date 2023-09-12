import os
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from datasets import load_from_disk
import torch
from peft import PeftConfig, PeftModel
from accelerate import Accelerator
import shutil


def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        default="tiiuae/falcon-7b",
        help="Model id to use for training.",
    )
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default="lm_dataset", help="Path to dataset.")
    # add training hyperparameters for epochs, batch size, learning rate, and seed
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for.")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    args = parser.parse_known_args()
    return args


def create_peft_config(model, gradient_checkpointing=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    
    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # prepare int-4 model for training
    model = prepare_model_for_kbit_training(model)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, peft_config)
    
    #this enables Naive Pipeline Processing, NPP
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    print(model.hf_device_map)
    
    model.print_trainable_parameters()
    return model


def training_function(args):
    # set seed
    set_seed(args.seed)

    dataset = load_from_disk(args.dataset_path)
    # load model from the hub with a bnb config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    #loading the model from path if not null otherwise use model_id
    model_to_load = args.model_path
    if model_to_load == "":
        model_to_load = args.model_id
    
    model = AutoModelForCausalLM.from_pretrained(
        model_to_load,
        use_cache=False if args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        trust_remote_code=True,  # ATTENTION: This allows remote code execution
        device_map="auto",
        #device_map={"": Accelerator().process_index},
        quantization_config=bnb_config,
    )

    # create peft config
    model = create_peft_config(model, args.gradient_checkpointing)

    # Define training args
    output_dir = "/tmp"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        bf16=args.bf16,  # Use BF16 if available
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        gradient_checkpointing=args.gradient_checkpointing,
        # logging strategies
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        ddp_find_unused_parameters=False, #important to work around that problem: https://github.com/artidoro/qlora/issues/137
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
    )

    # pre-process the model by upcasting the layer norms in float 32 for
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    # Start training
    trainer.train()

    if args.merge_weights:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(output_dir)
        # clear memory
        del model
        del trainer
        # load PEFT model in fp16
        peft_config = PeftConfig.from_pretrained(output_dir)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,  # ATTENTION: This allows remote code execution
        )
        model = PeftModel.from_pretrained(model, output_dir)
        model.eval()
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained("/opt/ml/model/")
    else:
        trainer.model.save_pretrained("/opt/ml/model/")

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.save_pretrained("/opt/ml/model/")

     # copy inference script
    os.makedirs("/opt/ml/model/code", exist_ok=True)
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "inference.py"),
        "/opt/ml/model/code/inference.py",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "requirements.txt"),
        "/opt/ml/model/code/requirements.txt",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "configuration_RW.py"),
        "/opt/ml/model/configuration_RW.py",
    )
    shutil.copyfile(
        os.path.join(os.path.dirname(__file__), "configuration_RW.py"),
        "/opt/ml/model/modelling_RW.py",
    )
    

def main():
    args, _ = parse_arge()
    training_function(args)


if __name__ == "__main__":
    main()
