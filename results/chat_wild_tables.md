
### probes, style=clean

| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |
|---|---|---|---|---|---|---|
| chat_slice @glm-5.3-flash | 95 | 100% | 520 | 1.0 | 0.0001 |  |
| chat_slice | 95 | 100% | 575 | 1.0 | 0.0002 |  |
| chat_terse | 95 | 100% | 799 | 1.0 | 0.0002 |  |
| chat_ledger0 | 95 | 100% | 1,375 | 1.0 | 0.0004 |  |
| chat_prune | 95 | 100% | 1,949 | 1.0 | 0.0006 |  |
| chat_useronly | 95 | 100% | 2,222 | 1.0 | 0.0006 |  |
| chat_asstclip | 95 | 100% | 4,024 | 1.0 | 0.0011 |  |
| chat_terse @glm-5.3-flash | 95 | 99% | 798 | 1.0 | 0.0001 | original 1 |
| chat_shorthand | 95 | 99% | 940 | 1.0 | 0.0002 | sum 1 |
| chat_tools0 | 95 | 99% | 2,112 | 2.4 | 0.0005 | sum 1 |
| chat_regex | 95 | 89% | 1,019 | 1.0 | 0.0003 | updated 4, sum 3, count 2, list 1 |
| chat_rag | 95 | 83% | 823 | 1.0 | 0.0002 | negation 5, sum 5, list 3, updated 2, absent 1 |

### probes, style=messy

| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |
|---|---|---|---|---|---|---|
| chat_useronly | 95 | 100% | 3,009 | 1.0 | 0.0008 |  |
| chat_slice | 95 | 99% | 514 | 1.0 | 0.0002 | updated 1 |
| chat_slice @glm-5.3-flash | 95 | 99% | 519 | 1.0 | 0.0001 | updated 1 |
| chat_ledger0 | 95 | 99% | 1,369 | 1.0 | 0.0004 | updated 1 |
| chat_useronly @glm-5.3-flash | 95 | 99% | 3,013 | 1.0 | 0.0003 | list 1 |
| chat_asstclip | 95 | 99% | 4,902 | 1.0 | 0.0013 | list 1 |
| chat_terse | 95 | 97% | 798 | 1.0 | 0.0002 | original 2, updated 1 |
| chat_shorthand | 95 | 97% | 985 | 1.0 | 0.0003 | list 1, early 1, sum 1 |
| chat_prune | 95 | 95% | 2,585 | 1.0 | 0.0007 | list 3, updated 2 |
| chat_regex | 95 | 80% | 1,307 | 1.0 | 0.0004 | updated 12, sum 3, early 2, count 1, list 1 |

### hand-off, style=clean

| worker | tasks | facts found | complete notes | stale values | prompt tok/task | $/task |
|---|---|---|---|---|---|---|
| worker_terse | 10 | 195/195 | 10/10 | 13 | 816 | 0.0005 |
| worker_slice | 10 | 195/195 | 10/10 | 13 | 914 | 0.0005 |
| worker_shorthand | 10 | 195/195 | 10/10 | 9 | 950 | 0.0005 |
| worker_useronly | 10 | 195/195 | 10/10 | 13 | 2,230 | 0.0010 |
| worker_regex | 10 | 185/195 | 5/10 | 16 | 1,031 | 0.0007 |

### hand-off, style=messy

| worker | tasks | facts found | complete notes | stale values | prompt tok/task | $/task |
|---|---|---|---|---|---|---|
| worker_useronly | 10 | 195/195 | 10/10 | 13 | 3,017 | 0.0012 |
| worker_terse | 10 | 194/195 | 9/10 | 14 | 814 | 0.0004 |
| worker_slice | 10 | 194/195 | 9/10 | 14 | 901 | 0.0005 |
| worker_shorthand | 10 | 194/195 | 9/10 | 9 | 996 | 0.0005 |
| worker_regex | 10 | 181/195 | 5/10 | 18 | 1,319 | 0.0007 |

### probes, style=clean

| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |
|---|---|---|---|---|---|---|
| chat_terse | 95 | 100% | 838 | 1.0 | 0.0002 |  |

### probes, style=messy

| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |
|---|---|---|---|---|---|---|
| chat_terse | 95 | 99% | 836 | 1.0 | 0.0002 | updated 1 |

### probes, style=clean

| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |
|---|---|---|---|---|---|---|
| chat_slice | 95 | 100% | 513 | 1.0 | 0.0002 |  |
| chat_rag | 95 | 86% | 533 | 1.0 | 0.0002 | sum 5, negation 4, updated 2, absent 1, list 1 |

### probes, style=clean

| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |
|---|---|---|---|---|---|---|
| chat_slice | 95 | 96% | 509 | 1.0 | 0.0002 | list 4 |

### probes, style=messy

| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |
|---|---|---|---|---|---|---|
| chat_slice | 95 | 100% | 515 | 1.0 | 0.0002 |  |
| chat_ledger0 | 95 | 100% | 1,386 | 1.0 | 0.0003 |  |
| chat_terse | 95 | 98% | 846 | 1.0 | 0.0002 | negation 2 |

### probes, style=messy

| condition | n | accuracy | prompt tok/q | LLM calls/q | $/q | failures by type |
|---|---|---|---|---|---|---|
| chat_summary | 38 | 100% | 4,366 | 1.0 | 0.0012 |  |
| chat_state | 38 | 97% | 4,755 | 1.0 | 0.0013 | updated 1 |
