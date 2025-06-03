import tiktoken


encoding = tiktoken.encoding_for_model("gpt-4o")
print(encoding)

# print(tiktoken.get_encoding("gpt-4o"))

# print(encoding.encode("Hello, world!"))

# print(encoding.decode(encoding.encode("Hello, world!")))