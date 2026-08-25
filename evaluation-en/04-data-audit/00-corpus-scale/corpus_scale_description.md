# BSRF-HEPS Corpus Scale

The BSRF-HEPS key-value corpus contains 550 complete K-V records, 472 unique keys, 480 unique values, and 547 unique K-V pairs. There are 3 exact duplicate rows (0.55%) and 25 keys associated with multiple values.

The key column contains 43,183 characters and the value column contains 188,787 characters. Together they contain 231,970 characters, or 199,345 after whitespace removal. A K-V record contains a mean of 421.76 characters and a median of 129.5; the range is 48-8935, with P90 and P95 values of 867.3 and 1597.45.

The corpus contains 0 Chinese characters, 32,775 English word units, and 1,831 numeric sequences. After excluding placeholder values such as "[Picture placeholder]" and "——", 527 substantive records remain (95.82% of complete records), containing 229,333 K-V characters.

## Summary Statistics

| Metric | Value |
|---|---:|
| Complete K-V records | 550 |
| Substantive K-V records | 527 |
| Placeholder records | 23 |
| Unique keys | 472 |
| Unique values | 480 |
| Unique K-V pairs | 547 |
| Exact duplicate K-V rows | 3 |
| Keys with multiple values | 25 |
| Key-column characters | 43,183 |
| Value-column characters | 188,787 |
| Total K-V characters | 231,970 |
| K-V characters excluding whitespace | 199,345 |
| Substantive K-V characters | 229,333 |
| Chinese characters | 0 |
| English word units | 32,775 |
| Mean K-V length | 421.76 characters |
| Median K-V length | 129.5 characters |
| K-V length P90 / P95 | 867.3 / 1,597.45 characters |

Note: total K-V characters are calculated by summing the key and value columns. Substantive records exclude placeholder values such as "[Picture placeholder]" and "——".
