mod llama;

use clap::Parser;
use llama::context::Context;
use llama::decoder::decode;
use llama::layers::{div, max, rand_sample, softmax};
use llama::param::HyperParam;
use llama::tokenizer::{Tokenizer, SOS};
use llama::weight::Weight;
use std::fs::File;
use std::io::Write;
use std::{io, time::Instant};

#[derive(Debug, Parser)]
struct Arg {
    /// Hyperparameter file path
    #[arg(long, short, default_value = "./model/stories110M/hp.yaml")]
    hp: String,

    /// Vocab file path
    #[arg(long, short, default_value = "./model/stories110M/vocab.bin")]
    vocab: String,

    /// Weight file path
    #[arg(long, short, default_value = "./model/stories110M/weight.bin")]
    weight: String,

    /// Maximum number of tokens to generate
    #[arg(short, short, long, default_value = "256")]
    max: usize,

    /// Temperature for sampling
    #[arg(short, short, long, default_value = "0.0")]
    temp: f32,

    /// initial Prompt string
    #[arg(short, short, long)]
    prompt: Option<String>,
}

fn main() -> io::Result<()> {
    let args = Arg::parse();
    let hp = HyperParam::from_yaml(File::open(args.hp)?)?;
    let vocab = Tokenizer::from_file(&hp, File::open(args.vocab)?)?;
    let weight = Weight::from_file(&hp, File::open(args.weight)?)?;

    let prompt = {
        let mut ret = vec![SOS];
        if let Some(prompt) = args.prompt {
            ret.extend(vocab.str_to_tok(prompt))
        }
        ret
    };

    println!("=== Model ===");
    println!("{:#?}", hp);

    let timer = Instant::now();
    {
        let mut ctx = Context::new(&hp);
        let mut toks: Vec<usize> = vec![];

        println!("=== Prompt ===");
        for (pos, tok) in prompt.iter().enumerate() {
            decode(*tok, pos, &hp, &mut ctx, &weight);
            print!("{:}", vocab.tok_to_str(*tok));
            std::io::stdout().flush().unwrap();
            toks.push(*tok);
        }
        println!();

        println!("=== Output ===");
        for pos in prompt.len()..args.max {
            let tok = *toks.last().unwrap();
            let logits = decode(tok, pos, &hp, &mut ctx, &weight);
            let next = if args.temp < 1e-5 {
                max(&logits).0
            } else {
                rand_sample(&softmax(&div(&logits, args.temp)))
            };
            print!("{:}", vocab.tok_to_str(next));
            std::io::stdout().flush().unwrap();
            toks.push(next);
        }
        println!();
    }
    let time = timer.elapsed().as_secs_f64();

    println!("=== Done ===");
    println!(" * {:.3} [s]", time);
    println!(" * {:.3} [tok/s]", args.max as f64 / time);

    Ok(())
}
