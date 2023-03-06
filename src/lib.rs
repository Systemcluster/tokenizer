use std::{cell::RefCell, collections::HashMap};

use serde::{Deserialize, Serialize};
use serde_with::{serde_as, Bytes};

mod tiktoken;
use tiktoken::*;

wit_bindgen::generate!("tokenizer");


#[derive(Debug)]
enum TokenizerVariant {
    TokenizerTiktoken(CoreBPE),
}

#[derive(Serialize, Deserialize, Debug)]
enum LoadTokenizerVariant {
    LoadTokenizerTiktoken {
        bpe:         String,
        special_bpe: Vec<(String, u32)>,
        regex:       String,
    },
}

#[derive(Serialize, Deserialize, Debug)]
struct LoadTokenizerInput {
    name:    String,
    variant: LoadTokenizerVariant,
}

#[derive(Serialize, Deserialize, Debug)]
struct EncodeInput {
    name:  String,
    input: String,
}

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
struct DecodeInput {
    name:  String,
    #[serde_as(as = "Bytes")]
    input: Vec<u8>,
}

thread_local! {
    static TOKENIZERS: RefCell<HashMap<String, TokenizerVariant>> = RefCell::new(HashMap::new());
}

struct TokenizerImpl;
impl Tokenizer for TokenizerImpl {
    fn load_tokenizer(input: Vec<u8>) -> Result<(), String> {
        let input = ciborium::de::from_reader::<LoadTokenizerInput, _>(&input[..])
            .map_err(|e| format!("{}: {:?}", e, input))?;
        match input.variant {
            LoadTokenizerVariant::LoadTokenizerTiktoken {
                bpe,
                special_bpe,
                regex,
            } => {
                let tokenizer = CoreBPE::new(
                    load_bpe(&bpe.as_bytes())?,
                    HashMap::from_iter(special_bpe.into_iter()),
                    &regex,
                )?;
                TOKENIZERS.with(|map| {
                    map.borrow_mut()
                        .insert(input.name, TokenizerVariant::TokenizerTiktoken(tokenizer))
                });
            }
        }
        Ok(())
    }

    fn encode(input: Vec<u8>) -> Result<Vec<u8>, String> {
        let input = ciborium::de::from_reader::<EncodeInput, _>(&input[..])
            .map_err(|e| format!("{}: {:?}", e, input))?;
        TOKENIZERS.with(|map| {
            let map = map.borrow();
            let tokenizer = map.get(&input.name).ok_or("Tokenizer not found")?;
            match tokenizer {
                TokenizerVariant::TokenizerTiktoken(tokenizer) => {
                    let result = tokenizer.encode(&input.input);
                    Ok(result.iter().map(|x| (*x as u32).to_le_bytes()).flatten().collect())
                }
            }
        })
    }

    fn decode(input: Vec<u8>) -> Result<Vec<u8>, String> {
        let input = ciborium::de::from_reader::<DecodeInput, _>(&input[..])
            .map_err(|e| format!("{}: {:?}", e, input))?;
        TOKENIZERS.with(|map| {
            let map = map.borrow();
            let tokenizer = map.get(&input.name).ok_or("Tokenizer not found")?;
            match tokenizer {
                TokenizerVariant::TokenizerTiktoken(tokenizer) => {
                    let tokens = input
                        .input
                        .chunks(4)
                        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let result = tokenizer.decode(&tokens);
                    Ok(result)
                }
            }
        })
    }
}

export_tokenizer!(TokenizerImpl);

#[cfg(test)]
pub mod test {
    use crate::tiktoken::*;
    use std::collections::HashMap;

    static CL100K: &[u8] = include_bytes!("../tests/cl100k_base.tiktoken");

    #[test]
    fn test_encode() -> Result<(), String> {
        pub fn load_special_bpe() -> HashMap<String, u32> {
            let mut tokens: HashMap<String, u32> = HashMap::new();
            tokens.insert("<|endoftext|>".to_string(), 100257);
            tokens.insert("<|fim_prefix|>".to_string(), 100258);
            tokens.insert("<|fim_middle|>".to_string(), 100259);
            tokens.insert("<|fim_suffix|>".to_string(), 100260);
            tokens.insert("<|endofprompt|>".to_string(), 100276);
            tokens.insert("<|im_start|>".to_string(), 100264);
            tokens.insert("<|im_end|>".to_string(), 100265);
            tokens
        }
        let tokenizer = CoreBPE::new(
            load_bpe(CL100K)?,
            load_special_bpe(),
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )?;

        let result1 = tokenizer.encode("Hello World!");
        let tokens1 = result1.iter().map(|x| (*x as u32)).collect::<Vec<_>>();
        println!("Tokens: {:?}", tokens1);
        assert_eq!(tokens1, &[9906, 4435, 0], "Tokens should be [9906, 4435, 0]");

        let result2 = tokenizer.encode("hello <|endoftext|>");
        let tokens2 = result2.iter().map(|x| (*x as u32)).collect::<Vec<_>>();
        println!("Tokens: {:?}", tokens2);
        assert_eq!(tokens2, &[15339, 220, 100257], "Tokens should be [15339, 220, 100257]");

        Ok(())
    }

    #[test]
    fn test_decode() -> Result<(), String> {
        pub fn load_special_bpe() -> HashMap<String, u32> {
            let mut tokens: HashMap<String, u32> = HashMap::new();
            tokens.insert("<|endoftext|>".to_string(), 100257);
            tokens.insert("<|fim_prefix|>".to_string(), 100258);
            tokens.insert("<|fim_middle|>".to_string(), 100259);
            tokens.insert("<|fim_suffix|>".to_string(), 100260);
            tokens.insert("<|endofprompt|>".to_string(), 100276);
            tokens.insert("<|im_start|>".to_string(), 100264);
            tokens.insert("<|im_end|>".to_string(), 100265);
            tokens
        }
        let tokenizer = CoreBPE::new(
            load_bpe(CL100K)?,
            load_special_bpe(),
            r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
        )?;

        let result1 = tokenizer.decode(&[9906, 4435, 0]);
        let string1 = String::from_utf8_lossy(&result1);
        println!("String: {:?}", string1);
        assert_eq!(string1, "Hello World!", "String should be \"Hello World!\"");

        let result2 = tokenizer.decode(&[15339, 220, 100257]);
        let string2 = String::from_utf8_lossy(&result2);
        println!("String: {:?}", string2);
        assert_eq!(string2, "hello <|endoftext|>", "String should be \"hello <|endoftext|>\"");

        Ok(())
    }
}
