use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use sha2::{Digest, Sha256};

const MAGIC: &[u8; 8] = b"DNWGTV0\0";
const DETWGT_VERSION: u32 = 2;
const DET_NUM_SPEC_VERSION: u32 = 1;
const PAYLOAD_ALIGNMENT: usize = 64;

#[derive(Debug)]
pub struct MmapDetwgt {
    mmap: Mmap,
    tensors: BTreeMap<String, TensorEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    pub dims: Vec<u64>,
    pub element_count: u64,
    pub element_width: u32,
    pub payload_offset: usize,
    pub payload_len: usize,
}

impl MmapDetwgt {
    pub fn open(path: &Path, expected_sha256: &str) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // SAFETY: The mapping is read-only and the file handle is not used for mutation here.
        let mmap = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        let actual = format!("{:x}", Sha256::digest(&mmap));
        if actual != expected_sha256 {
            bail!(
                "{} digest mismatch: manifest has {expected_sha256}, file has {actual}",
                path.display()
            );
        }
        let tensors = parse_directory(&mmap, path)?;
        Ok(Self { mmap, tensors })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorEntry> {
        self.tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("model artifact has no tensor '{name}'"))
    }

    pub fn values(&self, name: &str) -> Result<Vec<i32>> {
        let tensor = self.tensor(name)?;
        self.read_values(tensor, 0, tensor.element_count as usize)
    }

    pub fn row(&self, name: &str, idx: usize) -> Result<Vec<i32>> {
        let tensor = self.tensor(name)?;
        let cols = *tensor
            .dims
            .last()
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' has no dimensions"))?
            as usize;
        let start = idx
            .checked_mul(cols)
            .ok_or_else(|| anyhow::anyhow!("row {idx} offset overflows for tensor '{name}'"))?;
        self.read_values(tensor, start, cols)
            .with_context(|| format!("row {idx} out of range for tensor '{name}'"))
    }

    pub fn rows(&self, name: &str, start: usize, end: usize) -> Result<Vec<i32>> {
        let tensor = self.tensor(name)?;
        let cols = *tensor
            .dims
            .last()
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' has no dimensions"))?
            as usize;
        let count = end
            .checked_sub(start)
            .and_then(|rows| rows.checked_mul(cols))
            .ok_or_else(|| anyhow::anyhow!("rows {start}..{end} overflow for tensor '{name}'"))?;
        self.read_values(tensor, start * cols, count)
            .with_context(|| format!("rows {start}..{end} out of range for tensor '{name}'"))
    }

    pub fn column_slice(&self, name: &str, start: usize, end: usize) -> Result<Vec<i32>> {
        let tensor = self.tensor(name)?;
        let cols = *tensor
            .dims
            .last()
            .ok_or_else(|| anyhow::anyhow!("tensor '{name}' has no dimensions"))?
            as usize;
        if end > cols || start > end {
            bail!("columns {start}..{end} out of range for tensor '{name}' width {cols}");
        }
        let rows = tensor.element_count as usize / cols;
        let mut out = Vec::with_capacity(rows * (end - start));
        for row in 0..rows {
            let offset = row * cols + start;
            out.extend(self.read_values(tensor, offset, end - start)?);
        }
        Ok(out)
    }

    fn read_values(&self, tensor: &TensorEntry, start: usize, count: usize) -> Result<Vec<i32>> {
        let end = start
            .checked_add(count)
            .ok_or_else(|| anyhow::anyhow!("tensor value range overflows usize"))?;
        if end > tensor.element_count as usize {
            bail!(
                "tensor range {start}..{end} exceeds {} values",
                tensor.element_count
            );
        }
        let bytes_per_value = (tensor.element_width / 8) as usize;
        let byte_start = tensor.payload_offset + start * bytes_per_value;
        let byte_end = byte_start + count * bytes_per_value;
        let payload = self
            .mmap
            .get(byte_start..byte_end)
            .ok_or_else(|| anyhow::anyhow!("tensor payload range is outside mmap"))?;
        match tensor.element_width {
            16 => Ok(payload
                .chunks_exact(2)
                .map(|b| i16::from_le_bytes([b[0], b[1]]) as i32)
                .collect()),
            32 => Ok(payload
                .chunks_exact(4)
                .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()),
            other => bail!("unsupported detwgt element width {other}"),
        }
    }
}

fn parse_directory(bytes: &[u8], path: &Path) -> Result<BTreeMap<String, TensorEntry>> {
    let mut cursor = Cursor::new(bytes);
    if cursor.take(8)? != MAGIC {
        bail!("{} is not a detwgt artifact", path.display());
    }
    let version = cursor.u32()?;
    if version != DETWGT_VERSION {
        bail!(
            "{} is detwgt v{version}; this reader expects v{DETWGT_VERSION}",
            path.display()
        );
    }
    let spec = cursor.u32()?;
    if spec != DET_NUM_SPEC_VERSION {
        bail!("unsupported det_num spec version {spec}");
    }

    let tensor_count = cursor.u64()?;
    let mut tensors = BTreeMap::new();
    for _ in 0..tensor_count {
        let name_len = cursor.u32()? as usize;
        let name = String::from_utf8(cursor.take(name_len)?.to_vec())?;
        let rank = cursor.u32()? as usize;
        let dims = (0..rank)
            .map(|_| cursor.u64())
            .collect::<Result<Vec<_>>>()?;
        let element_count = cursor.u64()?;
        if element_count != dims.iter().product::<u64>() {
            bail!("tensor '{name}' element count does not match its shape");
        }
        let element_width = cursor.u32()?;
        if !matches!(element_width, 16 | 32) {
            bail!("tensor '{name}' has element width {element_width}");
        }
        let payload_len = cursor.u64()? as usize;
        let bytes_per_value = (element_width / 8) as usize;
        if payload_len != element_count as usize * bytes_per_value {
            bail!("tensor '{name}' payload length does not match its width");
        }
        let _max_row_mass = cursor.u64()?;
        cursor.align_to(PAYLOAD_ALIGNMENT, &name)?;
        let payload_offset = cursor.offset;
        cursor.take(payload_len)?;

        let entry = TensorEntry {
            dims,
            element_count,
            element_width,
            payload_offset,
            payload_len,
        };
        if tensors.insert(name.clone(), entry).is_some() {
            bail!("duplicate tensor '{name}' in artifact");
        }
    }
    if !cursor.at_end() {
        bail!("trailing bytes after the last tensor payload");
    }
    Ok(tensors)
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(len)
            .ok_or_else(|| anyhow::anyhow!("artifact offset overflows usize"))?;
        let slice = self
            .bytes
            .get(self.offset..end)
            .ok_or_else(|| anyhow::anyhow!("artifact ended mid-record"))?;
        self.offset = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn align_to(&mut self, alignment: usize, name: &str) -> Result<()> {
        let padding = (alignment - (self.offset % alignment)) % alignment;
        if self.take(padding)?.iter().any(|byte| *byte != 0) {
            bail!("tensor '{name}' has non-zero padding");
        }
        Ok(())
    }

    fn at_end(&self) -> bool {
        self.offset == self.bytes.len()
    }
}
