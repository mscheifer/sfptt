import re

s = """eee
aaa
bbb
ddd"""

print(re.split(r"aaa\n", s, maxsplit = 1))
print(re.split(r"\naaa", s, maxsplit = 1))
