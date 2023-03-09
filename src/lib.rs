use std::{cell::RefCell, collections::HashMap};

use serde::{Deserialize, Serialize};
use serde_with::{serde_as, BytesOrString};
use tokenizers::Tokenizer;

mod tiktoken;
use tiktoken::*;

wit_bindgen::generate!("tokenizer");


#[derive(Debug)]
enum TokenizerVariant {
    TokenizerTiktoken(CoreBPE),
    TokenizerHuggingface(Tokenizer),
}

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
#[serde(untagged)]
enum LoadTokenizerVariant {
    LoadTokenizerTiktoken {
        #[serde_as(as = "BytesOrString")]
        bpe:         Vec<u8>,
        special_bpe: Vec<(String, u32)>,
        regex:       String,
    },
    LoadTokenizerHuggingface {
        #[serde_as(as = "BytesOrString")]
        model: Vec<u8>,
    },
}

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
struct LoadTokenizerInput {
    #[serde_as(as = "BytesOrString")]
    name: Vec<u8>,
    data: LoadTokenizerVariant,
}

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
struct EncodeInput {
    #[serde_as(as = "BytesOrString")]
    name:           Vec<u8>,
    #[serde_as(as = "BytesOrString")]
    input:          Vec<u8>,
    special_tokens: Option<bool>,
}

#[serde_as]
#[derive(Serialize, Deserialize, Debug)]
struct DecodeInput {
    #[serde_as(as = "BytesOrString")]
    name:           Vec<u8>,
    #[serde_as(as = "BytesOrString")]
    input:          Vec<u8>,
    special_tokens: Option<bool>,
}

thread_local! {
    static TOKENIZERS: RefCell<HashMap<String, TokenizerVariant>> = RefCell::new(HashMap::new());
}

fn deserialize<T>(input: &[u8]) -> Result<T, String>
where
    T: serde::de::DeserializeOwned, {
    rmp_serde::from_slice(input).map_err(|e| format!("{:?}", e))
}

struct TokenizerImpl;
impl TokenizerInterface for TokenizerImpl {
    fn load_tokenizer(input: Vec<u8>) -> Result<u32, String> {
        let input = deserialize::<LoadTokenizerInput>(&input[..])?;
        match input.data {
            LoadTokenizerVariant::LoadTokenizerTiktoken {
                bpe,
                special_bpe,
                regex,
            } => {
                let tokenizer = CoreBPE::new(
                    load_bpe(&bpe)?,
                    HashMap::from_iter(special_bpe.into_iter()),
                    &regex,
                )?;
                TOKENIZERS.with(|map| {
                    map.borrow_mut().insert(
                        String::from_utf8(input.name).unwrap(),
                        TokenizerVariant::TokenizerTiktoken(tokenizer),
                    )
                });
            }
            LoadTokenizerVariant::LoadTokenizerHuggingface { model } => {
                let tokenizer = Tokenizer::from_bytes(&model).map_err(|e| format!("{:?}", e))?;
                TOKENIZERS.with(|map| {
                    map.borrow_mut().insert(
                        String::from_utf8(input.name).unwrap(),
                        TokenizerVariant::TokenizerHuggingface(tokenizer),
                    )
                });
            }
        }
        Ok(0)
    }

    fn unload_tokenizer(input: Vec<u8>) -> Result<u32, String> {
        let input = String::from_utf8(input).unwrap();
        TOKENIZERS.with(|map| {
            map.borrow_mut().remove(&input);
        });
        Ok(0)
    }

    fn encode(input: Vec<u8>) -> Result<Vec<u8>, String> {
        let input = deserialize::<EncodeInput>(&input[..])?;
        TOKENIZERS.with(|map| {
            let map = map.borrow();
            let tokenizer =
                map.get(&String::from_utf8(input.name).unwrap()).ok_or("Tokenizer not found")?;
            match tokenizer {
                TokenizerVariant::TokenizerTiktoken(tokenizer) => {
                    let result = tokenizer.encode(&String::from_utf8(input.input).unwrap());
                    Ok(result.iter().map(|x| (*x).to_le_bytes()).flatten().collect())
                }
                TokenizerVariant::TokenizerHuggingface(tokenizer) => {
                    let result = tokenizer
                        .encode(
                            String::from_utf8(input.input).unwrap(),
                            input.special_tokens.unwrap_or(true),
                        )
                        .map_err(|e| format!("{:?}", e))?;
                    Ok(result.get_ids().iter().map(|x| (*x).to_le_bytes()).flatten().collect())
                }
            }
        })
    }

    fn decode(input: Vec<u8>) -> Result<Vec<u8>, String> {
        let input = deserialize::<DecodeInput>(&input[..])?;
        TOKENIZERS.with(|map| {
            let map = map.borrow();
            let tokenizer =
                map.get(&String::from_utf8(input.name).unwrap()).ok_or("Tokenizer not found")?;
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
                TokenizerVariant::TokenizerHuggingface(tokenizer) => {
                    let tokens = input
                        .input
                        .chunks(4)
                        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    let result = tokenizer
                        .decode(tokens, !input.special_tokens.unwrap_or(false), true, true)
                        .map_err(|e| format!("{:?}", e))?;
                    Ok(result.into_bytes())
                }
            }
        })
    }
}

export_tokenizer_interface!(TokenizerImpl);


#[cfg(test)]
pub mod test {
    use crate::tiktoken::*;
    use std::{collections::HashMap, str::FromStr};
    use tokenizers::Tokenizer;

    static CL100K: &[u8] = include_bytes!("../tests/cl100k_base.tiktoken");
    static NEOX20B: &[u8] = include_bytes!("../tests/neox_20b_tokenizer.json");

    #[test]
    fn test_encode_tt() -> Result<(), String> {
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
    fn test_decode_tt() -> Result<(), String> {
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

    #[test]
    fn test_encode_hf() -> Result<(), String> {
        let tokenizer = Tokenizer::from_str(&String::from_utf8(NEOX20B.to_vec()).unwrap()).unwrap();

        let result1 = tokenizer.encode("Hello World!", true).unwrap();
        let tokens1 = result1.get_ids();
        println!("Tokens: {:?}", tokens1);
        assert_eq!(tokens1, &[12092, 3645, 2], "Tokens should be [12092, 3645, 2]");

        let result2 = tokenizer.encode("hello <|endoftext|>", true).unwrap();
        let tokens2 = result2.get_ids();
        println!("Tokens: {:?}", tokens2);
        assert_eq!(tokens2, &[25521, 209, 0], "Tokens should be [25521, 209, 0]");

        Ok(())
    }

    #[test]
    fn test_decode_hf() -> Result<(), String> {
        let tokenizer = Tokenizer::from_str(&String::from_utf8(NEOX20B.to_vec()).unwrap()).unwrap();

        let string1 = tokenizer.decode(Vec::from([12092, 3645, 2]), false, true, true).unwrap();
        println!("String: {:?}", string1);
        assert_eq!(string1, "Hello World!", "String should be \"Hello World!\"");

        let string2 = tokenizer.decode(Vec::from([25521, 209, 0]), false, true, true).unwrap();
        println!("String: {:?}", string2);
        assert_eq!(string2, "hello <|endoftext|>", "String should be \"hello <|endoftext|>\"");

        Ok(())
    }
}
