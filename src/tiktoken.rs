// Adopted from https://github.com/openai/tiktoken
// Adopted parts: Copyright (c) 2022 OpenAI, Shantanu Jain, MIT License

use std::{
    collections::{HashMap, HashSet},
    ops::Range,
    vec::Vec,
};

use base64::{alphabet, engine, Engine};
use bstr::ByteSlice;
use fancy_regex::Regex;

static BASE64: engine::GeneralPurpose =
    engine::GeneralPurpose::new(&alphabet::STANDARD, engine::general_purpose::PAD);

pub fn load_bpe(bpe: &[u8]) -> Result<HashMap<Vec<u8>, u32>, String> {
    let mut tokens: HashMap<Vec<u8>, u32> = HashMap::new();
    for line in bpe.split(|u| *u == '\n' as u8) {
        if line.len() == 0 {
            continue;
        }
        let (l, r) = line.split_once_str(" ").ok_or_else(|| "Invalid BPE".to_string())?;
        tokens.insert(
            BASE64.decode(&l).map_err(|e| e.to_string())?,
            r.as_bstr()
                .to_str()
                .map_err(|e| e.to_string())?
                .parse::<u32>()
                .map_err(|e| e.to_string())?,
        );
    }
    Ok(tokens)
}


#[derive(Debug)]
pub struct CoreBPE {
    encoder:                HashMap<Vec<u8>, u32>,
    special_tokens_encoder: HashMap<String, u32>,
    decoder:                HashMap<u32, Vec<u8>>,
    special_tokens_decoder: HashMap<u32, Vec<u8>>,
    regex:                  Regex,
    special_regex:          Regex,
    sorted_token_bytes:     Vec<Vec<u8>>,
}

impl CoreBPE {
    pub fn new(
        encoder: HashMap<Vec<u8>, u32>, special_tokens_encoder: HashMap<String, u32>, pattern: &str,
    ) -> Result<Self, String> {
        let regex = Regex::new(pattern).map_err(|e| e.to_string())?;

        let special_regex = {
            let _parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&_parts.join("|")).map_err(|e| e.to_string())?
        };

        let decoder: HashMap<u32, Vec<u8>> = encoder.iter().map(|(k, v)| (*v, k.clone())).collect();

        assert!(encoder.len() == decoder.len());

        let special_tokens_decoder: HashMap<u32, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        let mut sorted_token_bytes = encoder.keys().cloned().collect::<Vec<_>>();
        sorted_token_bytes.sort();

        Ok(Self {
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex,
            special_regex,
            sorted_token_bytes,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> { self._encode_native(text).0 }

    pub fn encode_with_unstable(&self, text: &str) -> (Vec<u32>, HashSet<Vec<u32>>) {
        self._encode_unstable_native(text)
    }

    pub fn decode(&self, tokens: &[u32]) -> Vec<u8> { self._decode_native(tokens) }

    fn _decode_native(&self, tokens: &[u32]) -> Vec<u8> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for token in tokens {
            let token_bytes =
                self.decoder.get(token).unwrap_or_else(|| &self.special_tokens_decoder[token]);
            ret.extend(token_bytes);
        }
        ret
    }

    fn _encode_ordinary_native(&self, text: &str) -> Vec<u32> {
        // This is the core of the encoding logic; the other functions in here
        // just make things complicated :-)
        let regex = &self.regex;
        let mut ret = vec![];
        for mat in regex.find_iter(text) {
            let piece = mat.unwrap().as_str().as_bytes();
            if let Some(token) = self.encoder.get(piece) {
                ret.push(*token);
                continue;
            }
            ret.extend(&byte_pair_encode(piece, &self.encoder));
        }
        ret
    }

    fn _encode_native(&self, text: &str) -> (Vec<u32>, u32) {
        let special_regex = &self.special_regex;
        let regex = &self.regex;
        let mut ret = vec![];

        let mut start = 0;
        let mut last_piece_token_len = 0;
        loop {
            let next_special;
            let start_find = start;
            next_special = special_regex.find_from_pos(text, start_find).unwrap();
            let end = next_special.map_or(text.len(), |m| m.start());

            // Okay, here we go, compare this logic to _encode_ordinary_native
            for mat in regex.find_iter(&text[start..end]) {
                let piece = mat.unwrap().as_str().as_bytes();
                if let Some(token) = self.encoder.get(piece) {
                    last_piece_token_len = 1;
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                last_piece_token_len = tokens.len() as u32;
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                    last_piece_token_len = 0;
                }
                None => break,
            }
        }

        // last_piece_token_len is how many tokens came from the last regex split. This is used
        // for determining unstable tokens, since you can't merge across (stable) regex splits
        (ret, last_piece_token_len)
    }

    fn _increase_last_piece_token_len(
        &self, tokens: Vec<u32>, mut last_piece_token_len: u32,
    ) -> (Vec<u32>, u32) {
        // Unfortunately, the locations where our regex splits can be unstable.
        // For the purposes of determining unstable tokens, unstable regex splitting
        // is only a problem if a split that was present disappears, since this can
        // lead to merging of tokens otherwise thought to be stable.
        // cl100k_base makes our life hard by including the \s*[\r\n]+
        // pattern. This can e.g. cause "\n" + " " to become "\n \n".
        // Here is a quick and dirty fix:
        {
            let token_is_all_space = |token| {
                self.decoder
                    .get(token)
                    .map(|token_bytes| {
                        token_bytes.iter().rev().all(|&b| [b' ', b'\n', b'\t'].contains(&b))
                    })
                    .unwrap_or(false)
            };
            if last_piece_token_len > 0
                && token_is_all_space(&tokens[tokens.len() - last_piece_token_len as usize])
            {
                while (last_piece_token_len < tokens.len() as u32)
                    && token_is_all_space(&tokens[tokens.len() - last_piece_token_len as usize - 1])
                {
                    last_piece_token_len += 1;
                }
            }
        }
        debug_assert!(last_piece_token_len <= tokens.len() as u32);

        (tokens, last_piece_token_len)
    }

    fn _encode_unstable_native(&self, text: &str) -> (Vec<u32>, HashSet<Vec<u32>>) {
        let (tokens, last_piece_token_len) = self._encode_native(text);
        if last_piece_token_len == 0 {
            // If last_piece_token_len is zero, the last token was a special token and we have
            // no unstable bytes
            return (tokens, HashSet::new());
        }
        let (mut tokens, last_piece_token_len) =
            self._increase_last_piece_token_len(tokens, last_piece_token_len);

        let unstable_bytes =
            self._decode_native(&tokens[tokens.len() - last_piece_token_len as usize..]);
        tokens.truncate(tokens.len() - last_piece_token_len as usize);

        // TODO: we should try harder to find additional stable tokens
        // This would reduce the amount of retokenising when determining completions
        // Refer to the logic in an older version of this file

        let mut completions = HashSet::new();
        if unstable_bytes.is_empty() {
            return (tokens, completions);
        }

        // This is the easy bit. Just find all single tokens that start with unstable_bytes
        // (including tokens that exactly match unstable_bytes)
        // Separating this from the loop below helps with performance in a common case.
        let mut point = self
            .sorted_token_bytes
            .partition_point(|x| x.as_slice() < unstable_bytes.as_slice());
        while point < self.sorted_token_bytes.len()
            && self.sorted_token_bytes[point].starts_with(&unstable_bytes)
        {
            completions.insert(vec![self.encoder[self.sorted_token_bytes[point].as_slice()]]);
            point += 1;
        }

        // Now apply even more brute force. At every (other) possible position for the straddling
        // token, concatenate additional bytes from that token (if any) to unstable_bytes,
        // and retokenise the whole thing and see what we get.
        for i in 1..unstable_bytes.len() {
            let prefix = &unstable_bytes[..i];
            let suffix = &unstable_bytes[i..];
            let mut point = self.sorted_token_bytes.partition_point(|x| x.as_slice() < suffix);
            // TODO: Perf optimisation if suffix starts with " "?
            while point < self.sorted_token_bytes.len()
                && self.sorted_token_bytes[point].starts_with(suffix)
            {
                let possibility = [prefix, self.sorted_token_bytes[point].as_slice()].concat();
                let encoded = match std::str::from_utf8(&possibility) {
                    // Morally, this is byte_pair_encode(&possibility, &self.encoder)
                    // But we might have introduced a regex split which would prevent merges.
                    // (particularly possible in the presence of unstable regex splits)
                    // So convert to UTF-8 and do regex splitting.
                    // E.g. with cl100k_base "  !" gets split to " " + " !",
                    // but byte_pair_encode("  !") != byte_pair_encode(" ")
                    Ok(s) => self._encode_ordinary_native(s),

                    // Technically, whether or not this arm is correct depends on whether there
                    // would be a regex split before the UTF-8 truncation point.
                    // Probably niche enough that no one will ever notice (after all, people didn't
                    // notice all the big holes in the previous unstable token implementation)
                    Err(_) => byte_pair_encode(&possibility, &self.encoder),
                    // Something like the following is intriguing but incorrect:
                    // Err(e) => self._encode_ordinary_native(unsafe {
                    //     std::str::from_utf8_unchecked(&possibility[..e.valid_up_to()])
                    // }),
                };
                let mut seq = Vec::new();
                let mut seq_len = 0;
                for token in encoded {
                    seq.push(token);
                    seq_len += self.decoder[&token].len();
                    if seq_len >= unstable_bytes.len() {
                        break;
                    }
                }
                completions.insert(seq);
                point += 1;
            }
        }

        // This is also not straightforward. While we generally assume that regex splits are stable,
        // unfortunately, they are not. That is, if adding bytes were to make a split appear in
        // unstable_bytes, this could make tokens possible which our logic would otherwise think
        // would be merged.
        // For example, with gpt2, the use of \s+(?!\S) means that "\n\n" could
        // develop a split, e.g. "\n\n0" splits into "\n"+"\n"+"0", making "\n" a possible token.
        // Here is a quick and dirty fix:
        // This isn't right if we ever remove \s+(?!\S)
        if unstable_bytes.len() > 1 {
            let last_decoded = bstr::decode_last_utf8(unstable_bytes.as_slice());
            if unstable_bytes.len() - last_decoded.1 > 0
                && last_decoded.0.map_or(false, |c| c.is_whitespace())
            {
                let mut reencoded = byte_pair_encode(
                    &unstable_bytes[..unstable_bytes.len() - last_decoded.1],
                    &self.encoder,
                );
                reencoded.extend(byte_pair_encode(
                    &unstable_bytes[unstable_bytes.len() - last_decoded.1..],
                    &self.encoder,
                ));
                completions.insert(reencoded);
            }
        }

        (tokens, completions)
    }
}

fn _byte_pair_merge<T>(
    piece: &[u8], ranks: &HashMap<Vec<u8>, u32>, f: impl Fn(Range<u32>) -> T,
) -> Vec<T> {
    // This is a vector of (start, rank).
    // The rank is of the byte pair starting at position start.
    // The rank of the last item in the vector is not a valid value.
    let mut parts: Vec<(u32, u32)> = (0..piece.len() as u32 + 1).map(|i| (i, u32::MAX)).collect();

    // NOTE: using a macro here because a closure fails to get inlined
    // according to optimization remarks.
    // A closure also cannot capture a reference to `piece` without
    // the borrow checker complaining about the mutable borrows during
    // the assignments later in this code.
    macro_rules! get_rank {
        ($start_idx:expr, $skip:expr) => {{
            let start_idx: usize = $start_idx;
            let skip: usize = $skip;
            if (start_idx + skip + 2) < parts.len() {
                ranks
                    .get(
                        &piece[parts[start_idx].0 as usize..parts[start_idx + skip + 2].0 as usize],
                    )
                    .map(|r| *r)
            } else {
                None
            }
        }};
        ($idx:expr) => {{ get_rank!($idx, 0) }};
    }

    // We look up the ranks once in the beggining and iteratively update
    // them during each merge, which reduces the number of rank lookups.
    for i in 0..parts.len() - 2 {
        match get_rank!(i) {
            Some(rank) => {
                // usize::MAX is a sentinel value and cannot be a valid rank
                debug_assert!(rank != u32::MAX);
                parts[i].1 = rank;
            }
            None => {
                continue;
            }
        };
    }

    // If you have n parts and m merges, this does O(mn) work.
    // We could do something with a heap and do O(m log n) work.
    // It is important to consider that n is often small (<100), and as such
    // the cache-locality benefits outweigh the algorithmic complexity downsides
    // of the `parts` vector data structure above.

    // Note that we hash bytes, not token pairs. As long as we train BPE the way we
    // currently do, this is equivalent. An easy way to break this would be to decouple
    // merge priority from token index or to prevent specific token merges.
    loop {
        if parts.len() == 1 {
            break;
        }

        // usize::MAX is a sentinel rank value allowing us to
        // take the min more quickly
        let mut min_rank: (u32, u32) = (u32::MAX, 0);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i as u32);
            }
        }

        if min_rank.0 != u32::MAX {
            let i = min_rank.1;

            // NOTE: We are about to remove parts[i + 1]. We do not do it
            // yet because there are cache-locality benefits to updating
            // parts[i] and parts[i-1] before removing, which could thrash
            // the cache. Thus, we update the rank calculation by skipping over
            // parts[i + 1], by invoking `get_rank!` with `skip = 1`.
            parts[i as usize].1 = get_rank!(i as usize, 1).unwrap_or(u32::MAX);
            if i > 0 {
                parts[i as usize - 1].1 = get_rank!(i as usize - 1, 1).unwrap_or(u32::MAX);
            }

            parts.remove(i as usize + 1);
        } else {
            break;
        }
    }
    let mut out: Vec<T> = Vec::with_capacity(parts.len() - 1);
    for i in 0..parts.len() - 1 {
        out.push(f(parts[i].0..parts[i + 1].0));
    }
    out
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>) -> Vec<u32> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(piece, ranks, |p| ranks[&piece[p.start as usize..p.end as usize]])
}
