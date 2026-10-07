# Pre-consumption opportunity attrition: identification and safe control

Date: 2026-10-07

Status: `FROZEN_BEFORE_OFFLINE_TRANSPORT_ANALYSIS`

## Scientific object

For assignment $i$, let $A_i\in\{0,1\}$ denote immediate release or the
registered five-second hold, let $Q_i\geq0$ be opportunity measured before
the hold begins, and let $D_i(a;s)$ indicate failure to deliver that
opportunity under assignment $a$ when the shared system operates at delay
saturation $s$. The tested experiments use approximately equal-mass
assignment, so the primary causal object is

\[
\theta_s=
\frac{E[Q_iD_i(1;s)]-E[Q_iD_i(0;s)]}{E[Q_i]}.
\]

This is a direct assignment effect under the tested saturation. It is not a
prevalence-invariant policy effect and does not identify spillovers at another
saturation.

## Proposition 1: consumed-only non-identifiability

Let a consumed-only record contain facts only for assignments with selection
indicator $C_i=1$, including their generation and consumption versions. No
functional of this record alone identifies $E[Q_iD_i]$ without assumptions
about assignments with $C_i=0$.

### Construction

Hold fixed every consumed assignment, its opportunity, and its consumption
version. In world W0, add no unconsumed positive-opportunity assignment. In
world W1, add an assignment with $C=0$, $Q=q>0$, and $D=1$. The
consumed-only records, and therefore every consumed-staleness statistic, are
identical in W0 and W1. Their total lost opportunities differ by $q$.
Consequently, consumed staleness cannot identify pre-consumption opportunity
attrition without an identifying model for the omitted candidates.

This proposition concerns observability, not correlation. Consumed age can be
correlated with M4 while remaining undefined for work that never trains.

## Proposition 2: randomized identification

Assume within the registered acquisition window:

1. $Q_i$ is measured before the assigned hold begins;
2. release assignment is randomized with known positive probability;
3. the observed terminal disposition equals the potential disposition under
   the assigned arm;
4. terminal disposition is observed, or bounded by the registered endpoint
   procedure; and
5. the saturation regime and system configuration are held fixed.

Then randomization identifies

\[
E[Q_iD_i(1;s)]-E[Q_iD_i(0;s)]
=E[Q_iD_i\mid A_i=1]-E[Q_iD_i\mid A_i=0].
\]

Normalization by the pooled pre-treatment mean of $Q_i$ identifies
$\theta_s$. Cross-fitted adjustment may improve precision but is not required
for identification. If interference is not summarized by the fixed saturation,
the result remains the assignment effect induced by the implemented
randomization, not an isolated no-interference effect.

## Proposition 3: exact-utility shield guarantee

At decision $t$, let the base scheduler choose batch $B_t$. Let
$\mathcal F(B_t)$ contain batches satisfying the registered cardinality,
service-band, overlap, liveness, and metadata requirements and exactly matching
each protected base utility $u_k(B_t)$. Because $B_t\in\mathcal F(B_t)$,
the feasible set is nonempty.

M4-Shield performs exact search over $\mathcal F(B_t)$, lexicographically
maximizing imminent M4, total M4, lower token count, and deterministic group
identity. It enacts the optimizer only for a strict imminent-M4 improvement;
otherwise it returns $B_t$. Therefore, whenever exact search completes:

- every protected utility is invariant: $u_k(S_t)=u_k(B_t)$;
- imminent M4 weakly dominates the base proposal;
- an enacted intervention has strictly greater imminent M4;
- the base proposal is returned when no strict improvement exists; and
- an error or unmet precondition fails closed to the base scheduler.

The guarantee applies only to explicitly protected utilities. It does not
guarantee preservation of unmeasured data properties or downstream quality.

## Control implication

Raw M4 opportunity is

\[
Q_i=\sum_j |A_{ij}|T_{ij},
\]

where $A_{ij}$ is the registered GRPO scalar advantage and $T_{ij}$ is the
valid actor-token count. A deployment score should separate the value of an
assignment from its causal risk of expiry:

\[
R_i(\delta\mid X_t)=
\widehat U_i(t)
\left[
\widehat P(D_i=1\mid do(A_i=1),X_t)
-\widehat P(D_i=1\mid do(A_i=0),X_t)
\right].
\]

The existing randomized ledgers can test transport of the risk component.
They cannot establish that $Q_i$, or any replacement $\widehat U_i$, is
true marginal learning utility. That construct requires an independently
frozen gradient-utility audit before a terminal-quality scheduler acquisition.
