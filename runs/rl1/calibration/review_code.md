# Calibration review: code

## ct-21-3-0 (ct|bit manipulation) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def find_single(nums):
    res = 0
    for n in nums:
        res ^= n
    return res
```
```

## ct-36-42-1 (ct|boolean checks and validation) rewards=[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0] reasons=['pass', 'pass', 'fail', 'fail', 'fail', 'pass', 'fail', 'pass']

```
```python
def is_valid_ranking(rankings):
    return sorted(rankings) == list(range(1, len(rankings) + 1))
```
```

## ct-18-41-4 (ct|sequences and series) rewards=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0] reasons=['timeout', 'fail', 'fail', 'pass', 'pass', 'pass', 'fail', 'pass']

```
```python
def last_term_within_budget(first, ratio, budget):
    if first > budget:
        return 0
    terms = 1
    term = first
    while term * ratio > budget:
        term *= ratio
        terms += 1
    return terms
```
```

## ct-3-10-4 (ct|list filtering) rewards=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0] reasons=['pass', 'fail', 'pass', 'pass', 'fail', 'pass', 'pass', 'pass']

```
```python
def odd_at_even_index(nums):
    return [x for i, x in enumerate(nums) if i % 2 == 0 and x % 2 != 0]
```
```

## ct-31-17-4 (ct|dates and times arithmetic) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def to_total_seconds(hours, minutes, seconds):
    return hours * 3600 + minutes * 60 + seconds
```
```

## ct-11-23-4 (ct|dictionaries) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def drop_extreme_lengths(shoe_sizes):
    if not shoe_sizes:
        return {}
    lo, hi = min(shoe_sizes.values()), max(shoe_sizes.values())
    return {k: v for k, v in shoe_sizes.items() if v != lo and v != hi}
```
```

## ct-37-37-1 (ct|math with fractions and gcd) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def are_equivalent(a, b):
    return a[0] * b[1] == a[1] * b[0]
```
```

## ct-31-42-3 (ct|dates and times arithmetic) rewards=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0] reasons=['pass', 'fail', 'pass', 'fail', 'pass', 'fail', 'pass', 'pass']

```
```python
def hours_minutes_label(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours}h {minutes}m"
```
```

## ct-18-0-2 (ct|sequences and series) rewards=[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0] reasons=['fail', 'fail', 'pass', 'fail', 'fail', 'fail', 'fail', 'fail']

```
```python
def first_isolated(s):
    if not s:
        return ""
    for i, c in enumerate(s):
        if i == 0 and c != s[i + 1]:
            return c
        if i == len(s) - 1 and c != s[i - 1]:
            return c
        if c != s[i - 1] and c != s[i + 1]:
            return c
    return ""
```
```

## ct-17-23-2 (ct|arithmetic and rounding) rewards=[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0] reasons=['fail', 'pass', 'pass', 'fail', 'pass', 'fail', 'fail', 'fail']

```
```python
def size_range_kb(lines):
    import math
    if not lines:
        return 0
    sizes = [int(line.split()[-1]) for line in lines]
    return math.ceil(max(sizes) - min(sizes)) // 1024
```
```

## ct-11-23-1 (ct|dictionaries) rewards=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0] reasons=['fail', 'fail', 'pass', 'fail', 'pass', 'pass', 'fail', 'pass']

```
```python
def area_extremes(areas):
    if not areas:
        return (None, None)
    sm = min(areas.values())
    la = max(areas.values())
    return (min(k for v, k in areas.items() if v == sm), min(k for v, k in areas.items() if v == la))
```
```

## ct-37-17-0 (ct|math with fractions and gcd) rewards=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'fail', 'fail', 'pass', 'pass']

```
```python
def compare_fractions(n1, d1, n2, d2):
    a, b = n1 * d2, n2 * d1
    return (a > b) - (a < b)
```
```

## ct-15-32-4 (ct|primes and divisors) rewards=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'harness_unsafe', 'pass', 'pass', 'pass']

```
```python
def filter_divisible(lines, d):
    result = []
    for line in lines:
        value = int(line.split("code=")[1].split()[0])
        if value % d == 0:
            result.append(line)
    return result
```
```

## ct-2-26-2 (ct|string formatting and parsing) rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] reasons=['pass', 'fail', 'fail', 'fail', 'fail', 'pass', 'fail', 'fail']

```
```python
def parse_accounting(s):
    s = s.strip()
    sign = -1 if s.startswith("(") else 1
    s = s.replace("(", "").replace(")", "").replace("$", "").replace(" ", "")
    float_part = s.replace(",", "")
    return sign * float(float_part)
```
```

## ct-6-34-3 (ct|sorting with custom keys) rewards=[1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'fail', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def sort_by_length_then_case(words):
    return sorted(words, key=lambda w: (len(w), w.lower()))
```
```

## ct-16-28-3 (ct|digits of integers) rewards=[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0] reasons=['fail', 'pass', 'pass', 'fail', 'pass', 'pass', 'fail', 'fail']

```
```python
def kth_line_number_digit(k):
    from itertools import islice
    return int(islice(''.join(str(n) for n in range(1, 200)), k, None)[0])
```
```

## ct-22-39-1 (ct|base conversion) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def total_hex_scores(scores):
    return {name: sum(int(s, 16) for s in values) for name, values in scores.items()}
```
```

## ct-36-20-3 (ct|boolean checks and validation) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def never_overdrawn(tran):
    total = 0
    for t in tran:
        total += int(t)
        if total < 0:
            return False
    return True
```
```

## ct-12-5-3 (ct|dictionaries of lists) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def group_even_odd(nums):
    result = {'even': [], 'odd': []}
    for n in nums:
        if n % 2 == 0:
            result['even'].append(n)
        else:
            result['odd'].append(n)
    return result
```
```

## ct-12-11-0 (ct|dictionaries of lists) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def first_last_events(schedule):
    result = {}
    for day, events in schedule.items():
        if events:
            result[day] = (events[0], events[-1])
    return result
```
```

## ct-26-19-0 (ct|sliding windows) rewards=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['fail', 'fail', 'harness_unsafe', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def count_windows_with_duplicates(codes, k):
    n = len(codes)
    return max(0, n - k + 1 - sum(b - a for a, b in zip(codes, codes[1:]) if a == b))
```
```

## ct-27-11-0 (ct|stacks and queues) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def reverse_string(s):
    stack = list(s)
    result = []
    while stack:
        result.append(stack.pop())
    return ''.join(result)
```
```

## ct-34-20-3 (ct|removing duplicates) rewards=[1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0] reasons=['pass', 'pass', 'pass', 'fail', 'fail', 'pass', 'fail', 'pass']

```
```python
def best_adjustments(entries):
    best = {}
    for name, adj in entries:
        if name not in best or adj > best[name]:
            best[name] = adj
    return list(best.items())
```
```

## ct-12-15-3 (ct|dictionaries of lists) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def group_even_odd(nums):
    result = {'even': [], 'odd': []}
    for n in nums:
        if n % 2 == 0:
            result['even'].append(n)
        else:
            result['odd'].append(n)
    return result
```
```

## oci-b389c7a10671 (oci|generic) rewards=[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'fail', 'pass', 'harness_unsafe', 'pass', 'pass', 'pass']

```
```python
def calculate_fibonacci(n):
    """
    Generate a Fibonacci sequence up to the nth number.

    Parameters:
    n (int): The number of terms in the Fibonacci sequence to generate.

    Returns:
    list: A list containing the Fibonacci sequence up to the nth number.
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    sequence = [0, 1]
    for i in range(2, n):
        next_number = sequence[-1] + sequence[-2]
        sequence.append(next_number)
    return sequence
```
```

## ct-13-26-3 (ct|sets) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
```python
def count_magnitudes(readings):
    return len({abs(r) for r in readings})
```
```

## oci-b558da254c71 (oci|generic) rewards=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0] reasons=['pass', 'fail', 'fail', 'fail', 'fail', 'fail', 'pass', 'pass']

```
```python
def calculate_total_cost(products):
    """
    Calculate the total cost for a list of products.

    Parameters:
    products (list of dict): A list of product dictionaries, each containing 'base_cost', 'additional_cost', and 'discount'.
    
    Returns:
    list of float: A list containing the total cost after applying additional costs and discounts for each product.
    """
    import math
    total_costs = []
    for product in products:
        base_cost = product.get('base_cost', 0)
        additional_cost = max(0, product.get('additional_cost', 0))
        discount = max(0, product.get('discount', 0))
        if discount > base_cost + additional_cost:
            discount = base_cost + additional_cost
        total_cost = base_cost + additional_cost - discount
        total_costs.append(total_cost)
    return total_costs
```
```

## ct-25-12-2 (ct|two-pointer techniques) rewards=[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0] reasons=['fail', 'fail', 'pass', 'pass', 'fail', 'pass', 'fail', 'pass']

```
```python
def min_crates(weights, limit):
    low, high = 0, len(weights) - 1
    total = 0
    while low <= high:
        if low < high and weights[low] + weights[high] <= limit:
            low += 1
        high -= 1
        total += 1
    return total
```
```

## ct-17-6-4 (ct|arithmetic and rounding) rewards=[1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'fail', 'pass', 'fail', 'pass', 'pass', 'pass']

```
```python
def round_up_to_multiple(n, m):
    return -(-n // m) * m
```
```

## ct-10-17-3 (ct|lists of tuples) rewards=[1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0] reasons=['pass', 'pass', 'pass', 'harness_unsafe', 'pass', 'pass', 'fail', 'fail']

```
```python
def swap_ends(tuples):
    return [t if len(t) <= 1 else (t[-1],) + t[1:-1] + (t[0],) for t in tuples]
```
```
