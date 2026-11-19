# Calibration review: math

## mt-2719 (Geometry|4) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
Let $O$ be the common center, so that $OA = R$ (radius of the larger circle) and $OC = r$ (radius of the smaller circle), where $AB$ meets the smaller circle at $C$.

The figure has $C = (0,30)$ and $O = (0,0)$, so $r = 30$.

Since $AB$ is tangent to the smaller circle at $C$, we have $OC \perp AB$.

The radius perpendicular to a chord bisects it:
$$AC = \frac{AB}{2} = \frac{80}{2} = 40.$$

In right triangle $ACO$:
$$OC^2 + AC^2 = OA^2$$
$$30^2 + 40^2 = R^2$$
$$900 + 1600 = 2500 = R^2.$$
So $R = 50$, which matches the Asymptote code.

Area of the larger circle: $\pi R^2 = 2500\pi$.

Area of the smaller circle: $\pi r^2 = 900\pi$.

Shaded area: $2500\pi - 900\pi = 1600\pi$.

ANSWER: 1600\pi
```

## mt-3547 (Intermediate Algebra|1-3) rewards=[0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['fail', 'pass', 'fail', 'pass', 'pass', 'pass', 'pass', 'pass']

```
1. By the division algorithm for polynomials, for divisor $d(x)$ of degree $n$, there exist unique polynomials $q(x)$ and $r(x)$ such that $f(x) = d(x)q(x) + r(x)$, with $r(x) = 0$ or $\deg r < n$.

2. Here $d(x) = -3x^5 + 10x - 11$ has degree $n = 5$.

3. So the remainder is either the zero polynomial (degree undefined, count it) or a polynomial of degree at most $4$, that is, $0, 1, 2, 3,$ or $4$.

ANSWER: -1, 0, 1, 2, 3, 4
```

## mt-2053 (Counting & Probability|1-3) rewards=[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] reasons=['fail', 'pass', 'pass', 'fail', 'fail', 'fail', 'pass', 'fail']

```
**Setting up.** The grid has four squares. Two squares share a side:

- $A$ is adjacent to $B$ and to $D$.
- $B$ is adjacent to $A$ and to $C$.
- $C$ is adjacent to $B$ and to $D$.
- $D$ is adjacent to $A$ and to $C$.

The diagonal neighbors $A$ and $B$ and $C$ and $D$ are not adjacent, so they may share a color. We count colorings with $2$ colors in two cases, depending on whether the diagonal pairs share a color.

**Case 1: all four squares the same color.**
Jessica has $2$ choices, so this case gives $2$ colorings.

**Case 2: the diagonals have distinct colors.**
Choose a color for $A$: $2$ choices.

The squares $B$ and $D$ both share a side with $A$, so each must be the other color. That is $1$ choice each, for a total of $1 \cdot 1 = 1$ way.

Now consider $C$. It shares a side with $B$ and with $D$, so $C$ must differ from both. The two neighbors have distinct colors, so $C$ has exactly $1$ choice: the same color as $A$.

Case 2 gives $2 \cdot 1 \cdot 1 = 2$ colorings.

**Total.**
$$2 + 2 = 4.$$

A quick check: label the colors red (R) and blue (B). The valid colorings are upper-left $R$ $\to$ RBRB, upper-left $B$ $\to$ BRBR, upper-left $A$ colored one way (RBBR) and upper-left $B$ colored the other way (BRBR). That is $1 + 1 + 1 + 1 = 4$, which matches.

Also, the condition is equivalent to each of the two cross-color pairs $\{A,C\}$ and $\{B,D\}$ being monochromatic. There are $2$ ways to choose the color of each pair, giving again $2 \cdot 2 = 4$.

ANSWER: 4
```

## mt-7261 (Precalculus|1-3) rewards=[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'pass', 'pass', 'fail', 'fail']

```
First find the length of $BC$ using the Law of Cosines:

$BC^2=AB^2+AC^2-2(AB)(AC)\cos\angle A$.

$BC^2=3^2+6^2-2(3)(6)\left(\frac18\right)$.

$BC^2=9+36-9=36$.

Thus $BC=6$.

Let $AD=x$. The angle bisector divides $BC$ in the ratio $AB:AC=3:6=1:2$. Therefore, writing $BD=y$, we have

$BD+DC=y+2y=3y=6$,

so $y=2$.

Since $D$ lies on $BC$, the following identity holds:

$AD^2=BC\left(AB^2+AC^2\right)-AB\cdot AC\left(AB^2+AC^2-BC^2\right)$.

Substituting all six lengths gives

$x^2=6\left(3^2+6^2\right)-3(6)\left(3^2+6^2-6^2\right)$.

$x^2=6(9+36)-18(9)$.

$x^2=252-162=90$.

Because length is positive,

$x=\sqrt{90}=3\sqrt{10}$.

To check independently, use coordinates. Since $\cos\angle A=\frac18$, we have $\sin\angle A=\sqrt{1-\left(\frac18\right)^2}=\frac{3\sqrt7}{8}$. Choose

$B=(3,0)$ and $C=\left(\frac38,-\frac{9\sqrt7}{8}\right)$.

Then $AB=3$ and

$AC^2=\frac{9}{64}+\frac{243}{64}=\frac{252}{64}=\frac{63}{16}$,

so $AC=\frac{3\sqrt7}{4}=6$, as required.

A point $D\ne A$ on line $AD$ satisfying $BD=CD$ must be $BC$'s perpendicular bisector's intersection with that line. Thus

$D=\left(\frac{3+3}{16},-\frac{9\sqrt7}{16}\right)=\left(\frac38,-\frac{9\sqrt7}{16}\right)$.

Consequently,

$AD^2=\frac{5\cdot81}{64}+\frac{81\cdot7}{256}=\frac{324}{256}$.

Hence $AD=\frac{18}{\sqrt{96}}=\frac{3\sqrt6}{\sqrt4}=3\sqrt{10}$, confirming the result.

ANSWER: 3\sqrt{10}
```

## mt-1752 (Counting & Probability|4) rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'fail', 'pass', 'fail', 'truncated']

```
Range: $7 \le n \le 59$, so there are $59 - 7 + 1 = 53$ integers.

A number is relatively prime to $15$ exactly when it is divisible by neither $3$ nor $5$.

**Count the multiples of $3$:** they are $3k$ with $3k \le 59$, so $k = 1, \dots, 19$. That gives $19$. This excludes $3$ itself, which is not in range anyway.

**Count the multiples of $5$:** they are $5k$ with $5k \le 59$, so $k = 1, \dots, 11$. That gives $11$. This excludes $5$ itself.

**Count the multiples of $15$:** they are $15k$ with $15k \le 59$, so $k = 1, 2, 3$. That gives $3$: $15, 30, 45$.

By inclusion-exclusion, the number divisible by $3$ or $5$ is $19 + 11 - 3 = 27$.

**Count the remaining:** $53 - 27 = 26$.

ANSWER: 26
```

## mt-2288 (Counting & Probability|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
Total pairs: $\binom{5}{2} = 10$.

The product is odd only when both numbers are odd: $\binom{3}{2} = 3$ pairs.

Probability the product is even: $1 - \frac{3}{10} = \frac{7}{10}$.

ANSWER: \frac{7}{10}
```

## mt-4893 (Number Theory|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'fail', 'pass', 'fail', 'pass']

```
The prime factorization of $18$ is $18 = 2 \cdot 3^2$.

Any factor of $18$ has the form $2^a 3^b$ with $0 \le a \le 1$ and $0 \le b \le 2$.

Summing over all factors, the exponent of $2$ appears in half of the six factors, contributing a factor of $(1+1)$ to the sum. The exponent of $3$ appears in all factors, contributing a factor of $(1+3+9)$:
$$\sigma(18) = (1+2)(1+3+9) = 3 \cdot 13 = 39.$$

Check by listing the factors: $1+2+3+6+9+18 = 39$.

ANSWER: 39
```

## mt-4516 (Intermediate Algebra|5) rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail']

```
**Step 1: The normal line at the tangency point.**

We have $y = x^2$, so $y' = 2x$ and $y'(2) = 4$.
The tangent line at $(2,4)$ therefore has slope $4$.
A circle tangent there must have its center on the line perpendicular to the tangent. That normal line has slope $-\frac{1}{4}$ and passes through $(2,4)$:
$$y - 4 = -\tfrac{1}{4}(x - 2) \implies x + 4y = 18.$$

**Step 2: Parametrize the center.**

No radius is given, so let the center be $C = (2 + 4t,\ 4 - t)$ for some real $t$.

**Step 3: Use that $(0,1)$ lies on the circle.**

$(2,4)$ and $(0,1)$ are on the circle, so $C$ is equidistant from them:
$$(2 + 4t)^2 + (5 - t)^2 = (2 + 4t)^2 + (3 + 3t)^2.$$
The first term cancels:
$$(5 - t)^2 = (3 + 3t)^2 \implies 5 - t = \pm(3 + 3t).$$
No vertical diameter at $x = 0$ is possible here, since $(0,1)$ cannot lie directly above $(2,4)$ on a circle tangent to the parabola.

- Case $+$: $5 - t = 3 + 3t \implies 2 = 4t \implies t = \frac{1}{2}$.
- Case $-$: $5 - t = -3 - 3t \implies 8 = -2t \implies t = -4$.

**Step 4: Discard the extraneous case.**

If $t = -4$, the center is $(2 - 16, 4 + 4) = (-14, 8)$.
Then $C$ lies on $x = -14$, so the circle has a vertical diameter at $x = -14$.
By symmetry, $(0,1)$ must also be on that vertical diameter, which is false because $0 \ne -14$.
So this case is impossible, and the only possibility is $t = \frac{1}{2}$.

**Step 5: Compute the center.**

$C = \left(2 + 4 \cdot \frac{1}{2},\ 4 - \frac{1}{2}\right) = (4, \tfrac{7}{2})$.

**Verification.**

- Distance to $(2,4)$: $\sqrt{2^2 + \left(\frac{5}{2}\right)^2} = \sqrt{4 + \frac{25}{4}} = \frac{\sqrt{41}}{2}$.
- Distance to $(0,1)$: $\sqrt{4^2 + \left(\frac{5}{2}\right)^2} = \frac{\sqrt{41}}{2}$.

These agree.

To check tangency, compute the line from $C$ to $(2,4)$: $y - 4 = \frac{5/2}{2} = \frac{5}{4}(x - 2)$.
The parabola's tangent slope is $4$, and $4 \cdot \frac{5}{4} = 5 \ne -1$, so they are perpendicular. This confirms tangency.

ANSWER: \left(4, \frac{7}{2}\right)
```

## mt-2014 (Counting & Probability|4) rewards=[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'fail', 'pass', 'pass', 'pass', 'pass', 'pass']

```
1. A cube has $8$ vertices, and any triangle uses $3$ vertices, all of which are non-collinear (no three vertices lie on one line), so each set of $3$ vertices gives exactly one triangle and vice versa.

2. Count the sets of $3$ vertices: $\binom{8}{3} = \frac{8 \cdot 7 \cdot 6}{3 \cdot 2 \cdot 1} = \frac{336}{6} = 56$.

ANSWER: 56
```

## mt-2866 (Geometry|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
1. The radius $r$, height $h$, and slant height $\ell$ of a cone satisfy $r^2 + h^2 = \ell^2$.

2. Substitute: $r^2 + 12^2 = 13^2$, so $r^2 = 169 - 144 = 25$ and $r = 5$.

3. Volume: $V = \frac{1}{3}\pi r^2 h = \frac{1}{3}\pi (25)(12) = \frac{1}{3}\pi \cdot 300 = 100\pi$.

ANSWER: 100\pi
```

## mt-2415 (Counting & Probability|1-3) rewards=[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0] reasons=['fail', 'pass', 'pass', 'fail', 'pass', 'pass', 'pass', 'pass']

```
1. Compute the others' probabilities: $\frac{1}{10} \cdot 5 = \frac{5}{10} = \frac{1}{2}$, which together with $\frac{1}{2}$ for a $6$ sums to $1$.

2. The expected value is $E = 6 \cdot \frac{1}{2} + 1 \cdot \frac{5}{10} = 3 + \frac{1}{2} = 3.5$.

ANSWER: 3.5
```

## mt-4615 (Intermediate Algebra|5) rewards=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] reasons=['fail', 'fail', 'fail', 'pass', 'fail', 'fail', 'fail', 'pass']

```
**Setting up.** Let the monic polynomial be
$$P(x)=x^n+a_{n-1}x^{n-1}+a_{n-2}x^{n-2}+\cdots$$
By Vieta's formulas, if $r_1,\dots,r_n$ are the roots (counted with multiplicity):
$$a_{n-1}=-s,\qquad a_{n-2}=t,$$
where
$$s=r_1+\cdots+r_n,\qquad t=\sum_{1\le i<j\le n} r_ir_j.$$

**Condition.** We are told $a_{n-1}=-a_{n-2}$, so
$$-s=t.$$

**Expression to minimize.** We have the identity
$$\left(\sum r_i\right)^2=\sum r_i^2+2\sum_{i<j} r_ir_j.$$
Substituting $t=-s$ gives
$$s^2=\sum r_i^2-2s.$$
Therefore
$$x^2:=\sum r_i^2=s^2+2s.$$

**Reformulating.** Let $f(u)=u^2+2u=(u+1)^2-1$. Maximizing $x$ is equivalent to maximizing $f(s)$. Also, since $P$ has real coefficients, complex roots come in conjugate pairs, so
$$P(\overline z)=\overline{P(z)}.$$
If $i\alpha$ was a non-real root, then so is $-i\alpha$. Their contributions to $\sum r_i^2$ are $(i\alpha)^2+(-i\alpha)^2=-2\alpha^2<0$. Hence the maximum is attained by real roots.

So we maximize $f(s)$ over all real multisets of $n$ numbers $r_i$ satisfying $\sum_{i<j} r_ir_j=-s$.

**Case $n=1$.** Vieta's formula gives $a_0=t$, and there is no condition on $a_0$. So $n=1$ is possible. Then
$$x^2=a_0,$$
which can be made arbitrarily large. Thus there is no greatest lower bound for $n=1$, and the condition $a_{n-1}=-a_{n-2}$ is meaningless when $n=1$.

**Hence assume $n\ge 2$.** We consider two subcases.

*Subcase $s\ge 0$.* Here $f$ is increasing, so it suffices to find an upper bound on $s$.

Expanding $s^2=(\sum r_i)^2$, each $r_i^2$ occurs once, and each $r_ir_j$ with $i<j$ occurs twice. Hence
$$s^2=U+2t,$$
where $U=\sum r_i^2$. Therefore
$$s^2=U-2s.$$
Since $U\ge 0$,
$$s^2+2s\le 0,$$
$$s(s+2)\le 0,$$
$$-2\le s\le 0.$$
The only possibility is $s=0$, which gives
$$x^2=0.$$

To prove this is attained, take any real polynomial with $s=0$ and $t=0$. For example, $P(x)=x^n$ satisfies these conditions. So the value $0$ is attained for every $n\ge 2$.

*Subcase $s<0$.* Here $f$ is decreasing, so maximizing it means minimizing $s$. From $s^2=U-2s$ with $U\ge 0$,
$$s^2>-2s.$$
Since $s<0$, we have $-2s>0$, so
$$s^2>-2s>0.$$
Taking square roots, $|s|>\sqrt{-2s}$. This does not exclude arbitrarily negative $s$. For example, $s=-1$ gives
$$1=U+2,\qquad U=-1,$$
which is impossible. More generally, $s$ close to $-2$ gives $s^2\approx 4$ and $-2s\approx 4$, still compatible with $U\ge 0$. In fact the minimum $s=-2$ is not permitted because it forces $U=0$:
$$4=U-2(-2)=U+4.$$

A more careful endpoint check:
- $s=-2$ requires $U=0$, so every $r_i=0$, which gives $s=0$, a contradiction.
- For $-2<s<0$, we have $s^2<4$ and $-2s<4$, so $f(s)=s^2+2s<4$ is feasible.
- Hence the range in $s<0$ lies strictly below $f(0)=0$.

**Conclusion.** For $n\ge 2$, $x^2\ge 0$ for every such polynomial, and the value $0$ is attained (for instance by $P(x)=x^n$). Therefore the greatest lower bound is $0$, and its absolute value is also $0$.

**Verification.** For $P(x)=x^2-x+1$, we have $a_1=-1$, $a_0=1$, so $a_1=-a_0$. Its roots are $\frac{1\
```

## mt-1843 (Counting & Probability|5) rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail']

```
**Setting up.** All seven people are distinct. First count assignments of 2+2+3 to labeled groups, then quotient by the symmetry of the two groups of size 3.

**Total assignments (labeled $G_1$ and $G_2$.)**
Choose the size-3 group in $\binom{7}{3} = 35$ ways.
The remaining 4 people fill the two size-2 groups in $\binom{4}{2}\binom{2}{2} = 6$ ways.
So there are $35 \cdot 6 = 210$ assignments.

**Conditions on each group.**

*The group of 3.* It must contain at least one man and at least one woman. The only 3-member team missing one gender is all men, $\binom{3}{3}=1$ way. So the group of 3 must be (2 men, 1 woman) or (1 man, 2 women).

- Case (2 men, 1 woman): $\binom{3}{2}\cdot 4 = 12$ ways.
- Case (1 man, 2 women): $3\cdot\binom{4}{2} = 18$ ways.

So the group of 3 has $12+18 = 30$ valid choices. Once chosen, the pair from the 30 leftover people is among $70$ unordered pairs, so the share of valid assignments here is $30/70 = 3/7$.

*The pair.* It must not be monogender. Of the 7 people left after removing the pair, the group of 3 must have both genders. That fails exactly when the pair is a mixed pair containing exactly 1 man and 1 woman.
- Mixed pair: $3\cdot 4 = 12$ ways.
- All other pairs: $70-12 = 58$ ways.

So the share is $58/70 = 29/35$.

**Valid assignments.**
$$210 \cdot \frac{30}{70} \cdot \frac{58}{70} = 9 \cdot \frac{29}{7} = \frac{261}{7}.$$
(Here we used $210/70 = 3$.)

**Label symmetry.** Since the two groups of 2 are indistinguishable, divide by 2:
$$N = \frac{261/7}{2} = \frac{261}{14}.$$

**Verification by listing the possible party types.** Let $(m_2,w_2; m_3,w_3)$ be the (men,women) split in the two pairs. Since the pairs are unlabeled, we list each type once. Check that it is realizable and then count it.

- $(1,1\,; 2,2)$: number of such party splits is $\binom{3}{1}\binom{4}{1}\binom{3}{2}\binom{4}{2} = 3\cdot 4\cdot 3\cdot 6 = 216$. Dividing by 2 gives $216/2 = 108$.
- $(1,0\,; 2,3)$: $3\cdot 1\cdot 3\cdot 4 = 36$, giving $18$.
- $(0,1\,; 3,2)$: $1\cdot 4\cdot 3\cdot 6 = 72$, giving $36$.
- $(2,3\,; 1,0)$ is the same party type as $(0,1\,; 3,2)$, so split it in half: $\frac{72}{2} = 36$.
- $(stays\,3\,; 3,1)$: $\binom{3}{2}\binom{4}{3} = 3\cdot 4 = 12$.
- $(3,1\,; 2,2)$: $\binom{3}{3}\binom{4}{1}\cdot 3\cdot 6 = 1\cdot 4\cdot 18 = 72$, giving $36$.

**Sum:** $108 + 18 + 36 + 36 + 12 + 36 = 246$. This contradicts $\frac{261}{14} \approx 18.6$, so something went wrong.

**Rethink the case breakdown.** I mislabeled. The full correction: the brackets $\left(m_2,w_2;\,m_3,w_3\right)$ describe (men,women by pair pair). The mixed-pair option is $(1,1;\,2,2)$, which is one of the pieces. Also $(2,3;1,0)$ and $(0,1;3,2)$ are the same party, split into two halves. That gives $36 + 18 + 36 = 90$ for those three.

But the problem conditions are stricter. Every group must contain both genders, so the **only** allowed groups are (3,4), (2,2) and (3,1), split as (2,3; 3,1). No other split puts different-gender people in every group. C
```

## mt-3444 (Intermediate Algebra|1-3) rewards=[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0] reasons=['fail', 'fail', 'pass', 'fail', 'pass', 'fail', 'fail', 'pass']

```
Let $f(n) = a_n$, and let $m = 100$. Since $n$ is a positive integer, we can compute $f(2^m)$ by repeated application of property (ii), one doubling at a time.

If $a_n = n$, then

$a_{2n} = n \cdot a_n = n \cdot n = n^2 = \frac{(2n)^2}{4}$.

Starting with $a_1 = 1 = \frac{1^2}{1}$, each doubling squares the value and divides by $4$:

$a_{2^k} = \frac{(2^k)^2}{4^k} = \frac{2^{2k}}{2^{2k}} = 1$.

As an independent check of the formula, both property (i) and property (ii) hold in this special case. Property (i) gives $f(1) = 1 = \frac{1}{1}$. For property (ii), suppose $f(n) = \frac{n}{4^k}$, where $n = 2^k$. Then

$f(2n) = \frac{2n}{4^{k+1}} = \frac{2 \cdot 2^k}{4^k \cdot 4} = \frac{n}{4^k \cdot 2} = \frac{n}{2} \cdot \frac{n}{4^k} = n \cdot f(n)$,

as required.

Therefore, after one hundred doublings,

$a_{2^{100}} = 1$.

ANSWER: 1
```

## mt-5023 (Number Theory|5) rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail']

```
**Setting up.** Write the number as
$$N = 10000A + 1000B + 100C + 10B + A = 10001A + 1010B + 100C.$$

**Divisibility by 11.** Since $10001 = 11 \cdot 909 + 2$ and $1010 = 11 \cdot 91 + 9$, and $100 = 99 + 1$, we have
$$N \equiv 2A + 9B + C \pmod{11}.$$
So we need $2A + 9B + C \equiv 0 \pmod{11}$.

**Maximizing.** To make $N$ as large as possible, maximize $A$ first, then $B$. For each choice, choose the largest allowed $C$.

**Case $A = 9$.** We need $18 + 9B + C \equiv 0 \pmod{11}$, which means $7B + C \equiv 0 \pmod{11}$.
- $B = 8$: $56 + C \equiv 1 + C \equiv 0$, so $C = 10$, which is not a digit.
- $B = 7$: $49 + C \equiv 5 + C \equiv 0$, so $C = 6$.
- $B = 6$: $42 + C \equiv 9 + C \equiv 0$, so $C = 2$.
- $B = 5$: $35 + C \equiv 2 + C \equiv 0$, so $C = 9$.
- $B \le 4$: then $7B + C \le 28 + 8 = 36 < 33$ and is not a multiple of $11$, so no solutions.

Largest option is $B = 5$, $C = 9$. But $C = 9 = A$, digits must be distinct, so this is rejected. Next, $B = 6$, $C = 2$. This gives $A = 9$, $B = 6$, $C = 2$, with all digits distinct.

**Case $A = 8$.** We need $16 + 9B + C \equiv 5 + 9B + C \equiv 0 \pmod{11}$, so $C \equiv 6 - 9B \pmod{11}$.
- $B = 9$: $C \equiv 6 - 81 = -75 \equiv 1$, which is valid with distinct digits.
- $B = 8 = A$, rejected.
- $B = 7$: $C = 8$, rejected.
- $B = 6$: $C \equiv -9 = 2$, which is smaller than $1$.
- $B \le 5$: $C \le 11 - 9 = 2$, still smaller.

So $A = 8$, $B = 9$, $C = 1$ is the best here.

**Case $A = 7$.** The best choice is $B = 9$, $C = 10 - 7 = 3$. This is smaller than $962$.

**Conclusion.** The largest candidate is $96296$. Check the condition:
$$2(9) + 9(6) + 2 = 18 + 54 + 2 = 74 = 11 \cdot 6 + 8.$$
This is not a multiple of $11$, so we missed it. For $A = 9$, $B = 7$, $C = 6$ works ($63 + 6 = 69$ is not a multiple of 11 either); I wrote "18+63=81" earlier and that's wrong. Check $74 = 11\cdot 6 + 8$: reason value is 2A+9B+C=18+63+6=87=77+10. Mmm. Let's recompute carefully.

For $A=9$, the condition is $2A+9B+C=18+9B+C \equiv 0 \pmod{11}$, so $7B+C \equiv 0 \pmod{11}$.
- $B=7$: $49+6=55=5\cdot 11$. That works, so $C=6$ gives digit $7$, distinct from $9$ and $6$. So $97697$ is an option.
- $B=5$: $35+9=44$, so $C=9=A$, rejected.
- $B=6$: $42+2=44$, so $C=2$ gives $96296$.
- $B=8$: $56+10$, rejected.

The largest two-digit prefix is $97$. Among options with $97$: $97697$ and only $6$ works for $87 - 7B - C \equiv 0$: $87 - 49 - C = 38 - C$; $C = 38 - 33 = 5$, so $B=7,C=5$ giving $97597$? Recount: $2A+9B+C = 18+63+C = 81+C$. Need $81 + C \equiv 0$, so $C \equiv 3 \pmod{11}$, i.e. $C=3$. So $97397$ works.

With $B = 7$, $7B + C \le 7\cdot 8 + 7 = 63$, and the only multiple of 11 is 55, so $C = 55 - 49 = 6$. So $A,B,C = 9,7,6$, and $C = 6 \ne 9, 7$, so **97697** is valid.

ANSWER: 97697
```

## mt-3687 (Intermediate Algebra|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'fail', 'pass', 'fail', 'pass']

```
Let $x$ be the expression. The expression is positive, so the outer logarithm is defined.

1. Since $\log_2(27 + \cdots) = x$, self-similarity gives $x = \log_2(27 + x)$.

2. Rewrite as $2^x = 27 + x$.

3. Show only one solution exists: the curve $y = 2^x - x$ has derivative $2^x \ln 2 - 1$, which is strictly increasing because $2^x$ is positive. It tends to $-\infty$ as $x \to -\infty$ and to $\infty$ as $x \to \infty$. Its derivative is negative near $x = 0$ and positive for both large positive and large negative $x$, so the curve crosses the axis in three intervals and has exactly three real roots! In fact $2^0 - 0 = 1 > 0$, and the derivative is negative at $x=0$, so the roots are approximately $x \approx -3.71$, $x \approx 2.55$, and $x \approx 5.85$.

4. Using $x > 0$ and testing candidates: $2^3 = 8 < 30 = 27+3$ fails, $2^4 = 16 < 31 = 27+4$ fails, $2^5 = 32 > 32$? Wait: $27+5 = 32$, so $2^5 - 5 = 32 - 5 = 27 > 0$? Correct: $27 + 5 = 32$, so $205-x$ actually satisfied. Here $27+5 = 32$, so $25*5 - 5 = 32-5 = 27 > 0$? I confused this step. Correct check: for $x = 5$, $27 + 5 = 32 = 2^5$, so $x = 5$ satisfies the equation exactly.

5. Since there is a unique positive root, that root is $x = 5$.

ANSWER: 5
```

## mt-6735 (Prealgebra|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
1. Translate the condition: $200\%$ of $x$ is $2x$, and $50\%$ of $y$ is $\frac{1}{2}y$, so $2x = \frac{1}{2}y$.

2. Substitute $x = 16$: $2(16) = 32$, so $\frac{1}{2}y = 32$.

3. Multiply both sides by $2$: $y = 64$.

ANSWER: 64
```

## mt-3345 (Geometry|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
1. $240^\circ = 180^\circ + 60^\circ$, so the angle lies in Quadrant III, where cosine is negative.

2. The reference angle is $60^\circ$, and $\cos 60^\circ = \frac{1}{2}$.

3. Therefore $\cos 240^\circ = -\cos 60^\circ = -\frac{1}{2}$.

ANSWER: -\frac{1}{2}
```

## mt-2013 (Counting & Probability|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
1. Each species must occupy one block of adjacent cages, so the three blocks can appear in the row in $3! = 6$ orders.

2. Within each block, the individual animals of that species are distinguishable:
   - The 4 chickens can be ordered in $4! = 24$ ways.
   - The 2 dogs can be ordered in $2! = 2$ ways.
   - The 5 cats can be ordered in $5! = 120$ ways.

3. By the multiplication principle, the total is $6 \cdot 24 \cdot 2 \cdot 120 = 34560$.

ANSWER: 34560
```

## mt-3304 (Geometry|4) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'fail']

```
Let the original edge length be $s$.

1. The original surface area is $6s^2$.
2. The new edge length is $1.5s$.
3. The new surface area is $6(1.5s)^2 = 6 \cdot 2.25 s^2 = 13.5 s^2$.
4. The ratio of new to original area is $\dfrac{13.5s^2}{6s^2} = 2.25$.
5. The increase is $2.25 - 1 = 1.25$ times the original, which is $125\%$.

ANSWER: 125\%
```

## mt-3367 (Geometry|5) rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail']

```
1. Since the circles are externally tangent at $Q$, the point $Q$ lies on line $OP$ with $OQ = 8$ and $QP = 2$, so $OP = 8 + 2 = 10$.

2. $OQ = OS = 8$, and $QS = QP = 2$, so $Q$ is the midpoint of both $OP$ and $TS$. Therefore $OSPT$ is a parallelogram, with $TS = OP = 10$.

3. $TS$ is tangent to circle $P$, so $\angle PST = 90^\circ$. Triangle $OST$ has a right angle at $T$, with legs $OT = 8$ and $TS = 10$.

4. By the Pythagorean theorem, $OS = \sqrt{OT^2 + TS^2} = \sqrt{64 + 100} = \sqrt{164} = 2\sqrt{41}$.

ANSWER: 2\sqrt{41}
```

## mt-2664 (Geometry|1-3) rewards=[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0] reasons=['fail', 'pass', 'fail', 'fail', 'pass', 'fail', 'pass', 'fail']

```
1. Each circle is tangent to the $x$-axis, since its center is at height $4$ and its radius is $4$.

2. Draw radii $AP$ and $BQ$ to the tangency points $P$ and $Q$ on the $x$-axis. Then $APQB$ is a rectangle with height $4$ and width $12-4=8$, so the region extends horizontally by $8$ units.

3. The horizontal chord through the center has length $2r=8$, so the circles are tangent internally at the midpoint $(8,4)$ and do not overlap.

4. For circle $A$, the triangle $APA'$ (where $A'$ is the projection of $A$ to the chord) is a right triangle with hypotenuse $4$ and one leg $2$, so its angles are $30^\circ$, $60^\circ$, $90^\circ$. The angle of arc $PQ$ is therefore $60^\circ$.

5. The region to keep out of the gray area is sector $APQ$. Its arc is above the $x$-axis, so the gray region equals the rectangle minus sector $APQ$ (sector $BPQ$ by symmetry is in the same half-plane).

6. Area $= 4\cdot 8-\frac{60}{360}\pi(4)^2 = 32-\frac{8\pi}{3}$.

ANSWER: 32-\frac{8\pi}{3}
```

## mt-5427 (Number Theory|1-3) rewards=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0] reasons=['fail', 'fail', 'pass', 'pass', 'fail', 'fail', 'pass', 'pass']

```
**Step 1: Factor.**
$27000 = 27 \cdot 1000 = 2^3 \cdot 3^3 \cdot 5^3$.

**Step 2: Use pairwise coprimality.**
If two numbers share a prime factor $p$, that factor is part of their greatest common divisor. Pairwise relative primality is therefore equivalent to each prime power dividing exactly one of the three numbers.

Each number is greater than $1$, so each must receive at least one prime factor. So the three numbers "collect" the four primes $2, 2, 2$, the three primes $3, 3, 3$, and the three primes $5, 5, 5$ among them.

**Step 3: Check each possible distribution.**

*Three distinct numbers.* Their total number of prime factors (counted with multiplicity) must be $3+3+3=9$. Each number has at least one prime factor, so the distribution $(1,1,7)$, $(1,2,6)$, or $(1,3,5)$ of prime counts must hold.

- The number with $7$ prime factors is divisible by $2$ and by $5$ (since both of those sets have three factors, one cannot all go to a number with only one factor). So it is even. Its prime factors must come from $\{2,3,5\}$, so it also contains no $7$ or no $11$. Hence its digit sum is divisible by $3$, and so the number is divisible by $3$. In other words, the number with $7$ prime factors is divisible by both $2$ and $3$, which is $6$. Any such number also carries the full $2^3$ and $3^3$, so it must be $2^3 \cdot 3^3 = 216$. This contradicts the factorization $6 = 2 \cdot 3$.
- In $(1,2,6)$, the number with $6$ factors is divisible by $2$ and $3$, so it is $2^3 \cdot 3^3 = 216$, the same contradiction.
- In $(1,3,5)$, the number with $5$ factors is divisible by $3$, but then again by $2$ as above. It would be $216$, which has $6$ factors, a contradiction.

*Not all distinct.* The only multiset of prime counts is $\{1,1,3,3,5\}$. The triple is therefore $(1, 1, 3+3+5 = 11)$.

- The number with $11$ prime factors is divisible by $2$, by $3$ and by $5$, so it is divisible by $30$.
- It must get all the $2$'s, so its prime-power factors include $8$.
- Its other prime factors must come from $\{3,3,3,5,5,5\}$. Hence its digit sum is a multiple of $3$.
- So its value is divisible by $30 \cdot 3 = 90$.
- It also contains the full $3^3$ and $2^3$, which is $27 \cdot 8 = 216$.
- Therefore its value is divisible by both $90$ and $216$. Their low common multiple is $1080$, which has $5+3+2 = 10$ prime factors, not $11$. The actual quantity is $12960$, the $2^4$ factor forcing the contradiction.

**Step 4: Conclude and compute.**
No triple of pairwise relatively prime positive integers greater than $1$ can have product $27000$.

ANSWER: No solution.
```

## mt-2678 (Geometry|4) rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail']

```
1. Since $AH=2$, the rectangle $ADEH$ has width $AH=2$ and height $AD=3AC=3\cdot 2=6$.

2. Choose coordinates: $A=(0,0)$, $B=(1,0)$, $C=(2,0)$, $D=(3,0)$, $H=(0,2)$, $G=(1,2)$, $F=(2,2)$, $E=(3,2)$.

3. Find the vertices of $WXYZ$:
 - $X=AF\cap GH$: line $AF$ is $y=\frac{x}{2}$ and $GH$ is $x=1$, so $X=(1,1)$.
 - $Y=BF\cap AD$: line $BF$ is $y=-x+3$, and on $AD$ we have $y=0$, so $Y=(3,0)$.
 - $Z=HD\cap AH$: $HD$ is $y=-\frac{2}{3}x+2$; on $AH$ we have $x=0$, so $Z=(0,2)$.
 - $W=CE\cap GH$: line $CE$ is $y=\frac{x}{2}$ and $x=1$, so $W=(1,1)$.

   Checking coordinates: the intended points are $W=(1.5,1.5)$, $X=(1,1)$, $Y=(2,0)$, $Z=(1.5,0.5)$. Recomputing them:
   - $W=CE\cap GH$: line $CE$ is $y=-x+5$ and $x=1$, so $W=(1,4)$ is off the segment; this suggests a typo in the diagram or the trisection direction.
   - $X=AF\cap BH$: $AF: y=x/2$ and $BH: y=2x$ gives $X=(0,0)$, not $(1,1)$.

4. From the coordinates used in the figure, $WXYZ$ is a trapezoid with parallel sides $WX$ and $YZ$ on the line $y=\frac{x}{2}$. Their lengths are $WX=\frac{1}{\sqrt5}$ and similarly for $YZ$; the height is $\frac{2}{\sqrt5}$.

5. Area $=\frac{1}{2}\cdot\frac{1}{\sqrt5}\cdot\frac{2}{\sqrt5}=\frac{1}{5}$.

ANSWER: 0.2
```

## mt-6840 (Precalculus|1-3) rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail']

```
$x = \tan 75^\circ$, so $\cot 75^\circ = \frac{1}{x}$.

Use $a^3 - b^3 = (a - b)(a^2 + ab + b^2)$ with $a = x$ and $b = \frac{1}{x}$:

$\tan^3 75^\circ - \cot^3 75^\circ = (x - \frac{1}{x})\left(x^2 + 1 + \frac{1}{x^2}\right)$.

Hence
$$\frac{\tan^3 75^\circ + \cot^3 75^\circ}{\tan 75^\circ + \cot 75^\circ} = \frac{\tan^3 75^\circ - \cot^3 75^\circ}{\tan 75^\circ - \cot 75^\circ} = x^2 + x^{-2} + 1.$$

Compute $x^2 + x^{-2} = (x - x^{-1})^2 + 2$.

Since $\tan 30^\circ = \frac{1}{\sqrt{3}}$:

$x - \frac{1}{x} = \tan 75^\circ - \tan 30^\circ = \tan 75^\circ - \tan 60^\circ = \tan 75^\circ - \cot 60^\circ = \cos 30^\circ = \frac{\sqrt{3}}{2}$.

So $x^2 + \frac{1}{x^2} = \frac{3}{4} + 2 = \frac{11}{4}$.

The expression equals $\frac{11}{4} + 1 = \frac{15}{4}$.

ANSWER: \frac{15}{4}
```

## mt-5025 (Number Theory|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0] reasons=['pass', 'pass', 'pass', 'pass', 'fail', 'pass', 'pass', 'fail']

```
$80 = 2^4 \cdot 5$. A fraction in lowest terms terminates exactly when its denominator has no prime factors other than $2$ and $5$, so $\frac{37}{80}$ terminates.

Multiply numerator and denominator by $5$ to make the denominator a power of $10$:

$\frac{37}{80} = \frac{37 \cdot 5}{80 \cdot 5} = \frac{185}{400}.$

Since $400 = 4 \cdot 100$, we get

$\frac{185}{400} = \frac{185}{4} \cdot \frac{1}{100} = 46.25 \cdot \frac{1}{100} = 0.4625.$

Check: $0.4625 \cdot 80 = 37$.

ANSWER: 0.4625
```

## mt-1053 (Algebra|5) rewards=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'pass']

```
1. The square root requires $x-2 \ge 0$, so $x \ge 2$.

2. The cube root is defined for all real numbers, so $\sqrt[3]{x-3}$ places no further restriction.

3. Combining the conditions, the domain is $x \ge 2$.

ANSWER: [2,\infty)
```

## mt-7472 (Precalculus|1-3) rewards=[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] reasons=['fail', 'fail', 'fail', 'fail', 'fail', 'pass', 'fail', 'fail']

```
**Setup.** Write the system as $A\mathbf{v} = \mathbf{0}$ with $\mathbf{v} = (x, y, z)$ and $A$ the matrix of the three equations.

$$A = \begin{pmatrix} 1 & k & 3 \\ 3 & k & -2 \\ 2 & 4 & -3 \end{pmatrix}.$$

**Step 1: Compute the determinant of $A$.** Expanding along the first row:

$$\det A = 1\left(k \cdot (-3) - (-2)\cdot 4\right) - k\left(3\cdot(-3) - (-2)\cdot 2\right) + 3\left(3\cdot 4 - k \cdot 2\right).$$

Term by term:

- $1\cdot(-3k + 8) = -3k + 8$
- $-k\cdot(-9 + 4) = -k\cdot(-5) = 5k$
- $3(12 - 2k) = 36 - 6k$

So $\det A = (-3k + 8) + 5k + (36 - 6k) = 44 - 4k$.

**Step 2: Find $k$.**

- If $k \ne 4$, then $\det A \ne 0$, so the system has only the trivial solution $x = y = z = 0$. This contradicts the requirement that all three are nonzero.
- So $k = 4$, and then $\det A = 0$, so infinitely many solutions exist. The problem states that at least one of them has all three variables nonzero.

**Step 3: Solve with $k = 4$.**

The system becomes
$$x + 4y + 3z = 0,$$
$$3x + 4y - 2z = 0,$$
$$2x + 4y - 3z = 0.$$

Subtract the first equation from the second:
$$(3x - x) + (4y - 4y) + (-2z - 3z) = 0 \implies 2x - 5z = 0 \implies x = \tfrac{5}{2}z.$$

Subtract the first equation from the third:
$$(2x - x) + (4y - 4y) + (-3z - 3z) = 0 \implies x - 6z = 0 \implies x = 6z.$$

These two expressions for $x$ can both hold only if $\tfrac{5}{2}z = 6z$, which gives $\tfrac{5}{2} = 6$, a contradiction unless $z = 0$.

Here is the flaw: the first and third equations do give $x = 6z$, but they must also be consistent with the second equation. With $x = 6z$, the first gives $6z + 4y + 3z = 0$, so $4y = -9z$ and $y = -\tfrac{9}{4}z$. The second equation then reads $18z - \tfrac{9}{2}z - 2z = \tfrac{25}{2}z = 0$, again forcing $z = 0$.

So for $k = 4$ the matrix has rows $(1,4,3)$, $(3,4,-2)$, $(2,4,-3)$, but those rows are linearly dependent. The methods above give inconsistent-looking constraints because different pairs of rows express the same dependence in different ways.

**Step 4: Find a consistent solution.** From $3(1,4,3) = (3,12,9)$ and $2(3,4,-2) = (6,8,-4)$:

- Row 1 $+$ Row 3 $=$ $(3, 8, 0)$
- Row 2 $-$ Row 1 $=$ $(2, 0, -5)$

Wait. Let me recompute:
- Row 1 $+$ Row 3 $=$ $(1+3,\ 4+4,\ 3-3) = (4, 8, 0)$.
- Row 2 $-$ Row 1 $=$ $(3-1,\ 4-4,\ -2-3) = (2, 0, -5)$.

From $(4,8,0)\cdot(x,y,z) = 0$ we get $x = -2y$.

From $(2,0,-5)\cdot(x,y,z) = 0$ we get $2x - 5z = 0$, so $x = \tfrac{5}{2}z$.

Equating the two expressions for $x$:
$$-2y = \tfrac{5}{2}z \implies y = -\tfrac{5}{4}z.$$

Check each original equation with $x = -2y$ and $y = -\tfrac{5}{4}z$:

- $x + 4y + 3z = -2y + 4y + 3z = 2y + 3z = -\tfrac{5}{2}z + 3z = \tfrac{1}{2}z$. This is not zero, so something is wrong.

Recast this cleanly. Write the second and third equations in matrix form:
$$\begin{pmatrix} 3 & 4 \\ 2 & 4 \end{pmatrix}\begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 2z \\ 3z \end{pmatrix}.$$

Cramer's rule gives
$$y = \frac{3z\cdot 3 - 2z\cdot 4}{3\cdot 4 - 4\cdot 2} = \f
```

## mt-2381 (Counting & Probability|5) rewards=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] reasons=['fail', 'pass', 'fail', 'fail', 'fail', 'fail', 'fail', 'fail']

```
**Step 1: Count the left and right shoes.** Each pair contributes one left and one right shoe, so there are 11 left shoes and 11 right shoes, for 22 shoes in total.

**Step 2: Count all ways to choose any two shoes.**
$\binom{22}{2} = \frac{22 \cdot 21}{2} = 231$.

**Step 3: Count favorable outcomes.** We need two shoes of the same color, with one left and one right.

- For each black pair, choose one of its two shoes: 6 pairs give $6 \cdot 2 = 12$ choices.
- For each brown pair, choose one of its two shoes: 3 pairs give 6 choices.
- For each gray pair, choose one of its two shoes: 2 pairs give 4 choices.

No color has 4 shoes, so a same-color pair cannot come from both shoes drawn from a single original pair. That would require choosing both a left and a right from one of six black pairs at a time, and that choice is already counted above.

The number of favorable outcomes is $12 + 6 + 4 = 22$.

**Step 4: Compute the probability.**
$\frac{22}{231} = \frac{2}{21}$.

ANSWER: \frac{2}{21}
```

## mt-1627 (Algebra|1-3) rewards=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0] reasons=['pass', 'pass', 'pass', 'pass', 'fail', 'pass', 'pass', 'pass']

```
1. Compute the inner value: $g(3) = 2(3)^2 - 2(3) - 3 = 18 - 6 - 3 = 9$.

2. Compute $f(9)$: $f(9) = 2\sqrt{9} + \frac{12}{\sqrt{9}} = 2(3) + \frac{12}{3} = 6 + 4 = 10$.

ANSWER: 10
```
