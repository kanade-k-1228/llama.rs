<div align="center">

# llama.rs

**LLaMA のシンプルな Rust 実装**

[English](README.md) | 日本語

</div>

```bash
$ git clone git@github.com:kanade-k-1228/llama.rs.git
$ cd llama.rs
$ cargo build --release
$ ./target/release/llama -p "Hello my name is"
```

## Raspberry Pi で実行

クロスコンパイラが無ければインストールします。

```bash
$ sudo apt update
$ sudo apt install gcc-aarch64-linux-gnu
```

Arm 向けのバイナリをコンパイルします。

```bash
$ cargo build --release --target aarch64-unknown-linux-gnu
```

バイナリとモデルファイルを転送します。

```bash
$ scp ./target/aarch64-unknown-linux-gnu/release/llama pi@?.?.?.?:/home/pi
$ scp -r model pi@?.?.?.?:/home/pi
```

RaspberryPi 上で実行します。

```bash
$ ssh pi@?.?.?.?
$ ./llama
```
