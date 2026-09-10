

Deep (Predictive) Discounted Counterfactual Regret Minimization
## Hang Xu
## 1,2
## , Kai Li
## 1,2,
## *
## , Haobo Fu
## 6
## , Qiang Fu
## 6
## , Junliang Xing
## 5
## , Jian Cheng
## 1,3,4
## 1
## C
## 2
DL, Institute of Automation, Chinese Academy of Sciences
## 2
School of Artificial Intelligence, University of Chinese Academy of Sciences
## 3
AiRiA
## 4
## Maicro.ai
## 5
## Tsinghua University
## 6
Tencent AI Lab
{xuhang2020, kai.li, jian.cheng}@ia.ac.cn,{haobofu, leonfu}@tencent.com, jlxing@tsinghua.edu.cn
## Abstract
Counterfactual regret minimization (CFR) is a family of algo-
rithms for effectively solving imperfect-information games.
To enhance CFR’s applicability in large games, researchers
use  neural  networks  to  approximate  its  behavior.  However,
existing methods are mainly based on vanilla CFR and strug-
gle to effectively integrate more advanced CFR variants. In
this  work,  we  propose  an  efficient  model-free  neural  CFR
algorithm,  overcoming  the  limitations  of  existing  methods
in approximating advanced CFR variants. At each iteration,
it collects variance-reduced sampled advantages based on a
value network, fits cumulative advantages by bootstrapping,
and applies discounting and clipping operations to simulate
the  update  mechanisms  of  advanced  CFR  variants.  Experi-
mental results show that, compared with model-free neural al-
gorithms, it exhibits faster convergence in typical imperfect-
information games and demonstrates stronger adversarial per-
formance in a large poker game.
Code— https://github.com/rpSebastian/DeepPDCFR
Extended version— https://arxiv.org/abs/2511.08174
## Introduction
Imperfect-information games (IIGs) serve as a foundational
framework for modeling strategic interactions among multi-
ple players where certain information remains hidden. Ad-
dressing these games poses significant challenges, as it re-
quires  reasoning  under  uncertainty  about  opponents’  pri-
vate  information.  Such  hidden  information  is  pervasive  in
real-world  scenarios,  such  as  negotiation  (Gratch,  Nazari,
and  Johnson  2016),  security  (Lisy,  Davis,  and  Bowling
2016), medical treatment (Sandholm 2015), and recreational
games (Brown and Sandholm 2019b), making research on
IIGs theoretically and practically crucial. The primary goal
in solving IIGs is to compute an (approximate) Nash equilib-
rium (NE) (Nash 1950)—a strategy profile where no player
can gain by unilaterally altering its strategy.
Similar  to  most  research  on  solving  IIGs,  we  focus  on
learning  an  NE  in  two-player  zero-sum  IIGs.  We  also  as-
sume the model-free setting, where the algorithm does not
## *
Corresponding author.
Copyright © 2026, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
have  an  exact  simulator  of  the  game  and  only  samples
episodes from the game. When the perfect game model is
available, the family of counterfactual regret minimization
(CFR) algorithms (Zinkevich et al. 2007; Tammelin 2014;
Brown and Sandholm 2019a; Farina, Kroer, and Sandholm
2021) is one of the most successful approaches for comput-
ing an NE, which iteratively minimizes the cumulative coun-
terfactual regrets of both players so that the average strat-
egy profile converges to an NE in two-player zero-sum IIGs.
Due to its robust theoretical foundation and strong empiri-
cal performance, CFR and its variants have driven several
significant advancements in this field (Bowling et al. 2015;
## Morav
## ˇ
c
## ́
ık et al. 2017; Brown and Sandholm 2018, 2019b).
When lacking a game model, outcome-sampling Monte
Carlo  CFR  (OS-MCCFR)  (Lanctot  et  al.  2009)  has  been
proposed to approximate the counterfactual regrets in each
iteration  by  sampling  episodes  from  the  game.  To  further
scale the algorithm to large IIGs, many novel neural CFR
variants have been developed. OS-DeepCFR (Brown et al.
2019) employs function approximation with deep neural net-
works to approximate the cumulative counterfactual regrets
instead of tabular storage. DREAM (Steinberger, Lerer, and
Brown  2020),  which  is  built  upon  variance-reduced  MC-
CFR (Schmid et al. 2019), employs a learned value function
as a baseline to reduce the high variance in estimating cumu-
lative counterfactual regrets. ESCHER (McAleer et al. 2023)
uses a history value function and a fixed sampling strategy
for the updating player to avoid using importance sampling.
Although  these  neural  approaches  have  greatly  acceler-
ated  CFR  in  large  IIGs,  they  primarily  focus  on  approx-
imating  the  behavior  of  vanilla  CFR  or  the  variant  Lin-
earCFR  (Brown  and  Sandholm  2019a).  Besides,  they  rely
on a large replay buffer to store experiences, which is used
to refit the cumulative counterfactual regrets each iteration.
Recent advancements in tabular CFR have demonstrated that
novel CFR variants can achieve significantly faster conver-
gence compared to vanilla CFR and LinearCFR. To reduce
the cost of picking wrong actions, CFR+ (Tammelin 2014)
clips negative cumulative counterfactual regrets in each it-
eration, Discounted CFR (DCFR) (Farina, Kroer, and Sand-
holm 2021) discounts the cumulative counterfactual regrets
in each iteration, and DCFR+ (Hang et al. 2022) combines
The Fortieth AAAI Conference on Artificial Intelligence (AAAI-26)
## 17284

the key insights of CFR+ and DCFR to achieve faster con-
vergence.  Predictive  CFR+  (PCFR+)  (Farina,  Kroer,  and
Sandholm  2021)  leverages  the  predictability  of  the  coun-
terfactual regrets in each iteration to accelerate the conver-
gence speed. PDCFR+ (Hang et al. 2024) integrates PCFR+
and DCFR in a principled manner, showcasing faster con-
vergence in non-poker IIGs. These tabular variants provide
a promising avenue for neural CFR to further unlock its po-
tential in achieving faster convergence.
To   this   end,   we   propose   novel   model-free   neural
CFR  variants,  Variance  Reduction  Deep  DCFR+  (VR-
DeepDCFR+) and Variance Reduction Deep PDCFR+ (VR-
DeepPDCFR+). These algorithms employ deep neural net-
works to approximate the behavior of DCFR+ and PDCFR+,
respectively, while leveraging learned baseline functions to
mitigate the high variance caused by episode sampling.
Approximating advanced CFR variants presents a signifi-
cant challenge, as these tabular CFR variants update cumu-
lative counterfactual regrets by bootstrapping from the pre-
vious iteration. This makes it impossible to directly approxi-
mate them using samples from all iterations stored in the re-
play buffer, as is typically done when approximating vanilla
CFR or LinearCFR. Furthermore, approximating the coun-
terfactual regrets in each iteration is particularly challeng-
ing since their computation depends on expectation values
weighted by the opponent’s reach probabilities. These un-
normalized values introduce substantial difficulties for neu-
ral networks to approximate effectively. Advantages, how-
ever, have been shown to be effectively approximated and
generalized using neural networks (Schulman et al. 2017),
and can be interpreted as a form of weighted counterfactual
regrets (Srinivasan et al. 2018). Therefore, we directly ap-
proximate  cumulative  advantages  during  each  iteration  by
using  the  samples  collected  in  the  current  iteration  while
bootstrapping from the results of the previous iteration.
Moreover,  under  the  model-free  setting,  the  algorithm
samples only a single action per state, resulting in high vari-
ance in the collected samples during each iteration. To mit-
igate this issue, we introduce an auxiliary value network in-
spired by DREAM (Steinberger, Lerer, and Brown 2020) to
reduce the variance induced by episode sampling. Building
upon the approximated cumulative advantages, we integrate
the novel tabular variants DCFR+ and PDCFR+ into neu-
ral CFR. For DCFR+, we apply discounting and clipping to
the  cumulative  advantages  at  each  iteration,  enabling  effi-
cient action reuse and controlling the potential overgrowth
of cumulative advantages over iterations. For PDCFR+, we
further introduce an additional network to fit the advantages
at each iteration, which is then used to predict the next ad-
vantages and compute the new strategy. Experimental results
demonstrate that these algorithms achieve competitive per-
formance compared to other model-free neural methods.
## Preliminaries
## Notations
Extensive-form games (Osborne and Rubinstein 1994) pro-
vide a tree-based formalism widely used to describe IIGs.
These games involve a finite setN={1,2,...,N}ofplay-
ers, along with a special playerccalledchancefollowing a
fixed, known stochastic strategy. Ahistoryhrepresents a se-
quence of all actions taken by players, including any private
information available only to specific players. The set of all
possible histories formsH, whileZ ⊆ Hdenotesterminal
historieswhere no further actions are possible. The set of
terminal histories that can be reached from a historyhis de-
noted byZ[h]  ={z∈Z:h⊑z}, where the relationship
g⊑hindicates thatgis equal to or aprefixofh. At any
given historyh, players choose from theactionsavailable,
represented byA(h) ={a:ha∈ H}. The player making
the decision at historyhis denoted byP(h). Each player
i∈Nis associated with autility functionu
i
(z) :Z →R,
which assigns a utility to each terminal historyz∈Z.
In IIGs, the lack of information is modeled usinginfor-
mation setsI
i
for each playeri∈ N, whereh,h
## ′
## ∈ Iin-
dicates that playericannot distinguish between them based
on the information available. For instance, in poker, histories
in  the  same  information  set  differ  only  in  opponents’  pri-
vate cards. Hence,A(I) =A(h)andP(I) =P(h)for any
h∈I. The set of all terminal histories that can be reached
from an information setIis denoted byZ[I] =
## S
h∈I
## Z[h],
andz[I]is the unique historyh∈Isuch thath⊑z.
## Astrategyσ
i
(I)assigns a probability distribution over
actionsA(I)available to playeriin information setI, where
σ
i
(I,a)represents the probability of playerichoosing ac-
tionainI. Similarly, the strategies must be consistent across
all histories in an information set. Thus, for anyh
## 1
## ,h
## 2
## ∈I,
we  haveσ
i
(I)  =σ
i
## (h
## 1
## )  =σ
i
## (h
## 2
).  Astrategy  profile
σ={σ
i
## |σ
i
## ∈Σ
i
,i∈ N}specifies the strategies for all
players, whereΣ
i
denotes the set of all possible strategies
for playeri, andσ
## −i
refers to the strategies of other players.
Thehistory reach probabilityπ
σ
(h)is the joint prob-
ability  of  reaching  historyhunder  strategy  profileσ,
computed  asπ
σ
## (h)   =
## Q
h
## ′
a⊑h
σ
## P(h
## ′
## )
## (h
## ′
,a).  It  factor-
izes  asπ
σ
## (h)   =π
σ
i
## (h)π
σ
## −i
(h),  whereπ
σ
i
(h)is  player
i’s  contribution,  andπ
σ
## −i
(h)is  the  contributions  of  all
other   players.   Theinformation   set   reach   probability
π
σ
(I)is  defined  asπ
σ
## (I)  =
## P
h∈I
π
σ
(h),  and  thein-
terval  information  set  reach  probabilityfromhtoh
## ′
is  defined  asπ
σ
## (h,h
## ′
## )   =π
σ
## (h)/π
σ
## (h
## ′
## )ifh
## ′
## ⊑h.
π
σ
i
(I),π
σ
## −i
(I),π
σ
i
## (h,h
## ′
## ),π
σ
## −i
## (h,h
## ′
)are defined similarly.
Theexpected utilityu
i
## (σ
i
## ,σ
## −i
)for playeridenotes her
utility when playingσ
i
against opponents’ strategyσ
## −i
## . For-
mally,u
i
## (σ
i
## ,σ
## −i
## ) =
## P
z∈Z
π
σ
## (z)u
i
(z). At the level of an
information setI, the expected utility for taking actionais
u
σ
i
(I,a) =
## P
h∈I
π
σ
## (h)
## P
z∈Z[ha]
π
σ
## (ha,z)u
i
## (z)
## P
h∈I
π
σ
## (h)
## ,
andforthewholeinformationset:u
σ
i
## (I)=
## P
a∈A(I)
σ
i
(I,a)u
σ
i
(I,a).   TheadvantageA
σ
i
(I,a)    =
u
σ
i
(I,a)−u
σ
i
(I)then quantifies the utility gain of playing
actionainstead of the current strategyσ
i
(I,a).
Best Response and Nash Equilibrium
Thebest responsetoσ
## −i
is any strategy BR(σ
## −i
)that max-
imizes the expected utility, satisfyingu
i
(BR(σ
## −i
## ),σ
## −i
## ) =
max
σ
## ′
i
## ∈Σ
i
u
i
## (σ
## ′
i
## ,σ
## −i
).  AnNEis  a  strategy  profileσ
## ∗
## =
## 17285

##  
σ
## ∗
i
## ,σ
## ∗
## −i
## 
where each player plays a best response to the oth-
ers, ensuring∀i∈N,u
i
## (σ
## ∗
i
## ,σ
## ∗
## −i
) = max
σ
## ′
i
## ∈Σ
i
u
i
## (σ
## ′
i
## ,σ
## ∗
## −i
## ).
Theexploitabilityof a strategyσ
i
is defined ase
i
## (σ
i
## )  =
u
i
## (σ
## ∗
i
## ,σ
## ∗
## −i
## )−u
i
## (σ
i
,BR(σ
i
)). In anε-NE, no player’s ex-
ploitability exceedsε. The exploitability of a strategy profile
σ, given bye(σ)  =
## P
i∈N
e
i
## (σ
i
)/|N|, represents the ap-
proximation error relative to the NE.
## Counterfactual Regret Minimization
Counterfactual Regret Minimization (CFR) (Zinkevich et al.
2007) frequently usescounterfactual valuev
σ
i
(I,a), which
is the expected utility of an actionaat an information setI∈
## I
i
for playeri, weighted by the probability thatireachesI.
## Formally,v
σ
i
(I,a) =
## P
z∈Z[I]
π
σ
## −i
(z[I])π
σ
(z[I]a,z)u
i
## (z),
andv
σ
i
## (I) =
## P
a∈A(I)
σ
i
(I,a)v
σ
i
(I). Starting from a uni-
form strategyσ
## 1
, CFR traverses the game tree in each iter-
ationtto computes theinstantaneous counterfactual re-
gretr
t
i
(I,a) =v
σ
t
i
(I,a)−v
σ
t
i
(I), which relates to the ad-
vantages asr
t
i
(I,a)  =A
σ
t
i
(I,a)π
σ
t
## −i
(I)(Srinivasan et al.
2018), accumulates it intocumulative counterfactual re-
gretR
t
i
(I,a)  =
## P
t
k=1
r
k
i
(I,a), and updates strategies by
regret-matching (Hart and Mas-Colell 2000):σ
t+1
i
(I,a) =
max(0,R
t
i
(I,a))
## P
a
## ′
## ∈A(I)
max(0,R
t
i
(I,a
## ′
## ))
, using the uniform strategy when all
regrets are non-positive. Theaverage strategy ̄σ
t
converges
to an NE and is computed viacumulative strategyC
t
i
(I,a):
## C
t
i
## (
## I
## ,a)=
## P
t
k=1
## 
π
σ
k
i
(I)σ
k
i
(I,a)
## 
## , ̄σ
t
i
(I,a)=
## C
t
i
(I,a)
## P
a
## ′
## ∈A
## (
## I)
## C
t
i
(I,a
## ′
## )
## .
Tabular CFR Variants
Since the birth of CFR, numerous variants have been pro-
posed  to  accelerate  convergence.CFR+(Tammelin  2014;
Bowling  et  al.  2015)  improves  convergence  by:  (1)  clip-
ping negative cumulative counterfactual regrets:R
t
i
(I,a) =
max(R
t−1
i
(I,a) +r
t
i
(I,a),0); (2) using a linear weighted
average strategy:C
t
i
(I,a) =C
t−1
i
(I,a) +tπ
σ
t
i
(I)σ
t
i
(I,a);
(3) applying alternating updates.DCFR(Brown and Sand-
holm 2019a) further introduces discounting:
## R
t
i
## (I
## ,a) =
## (
## R
t−1
i
(I,a)
## (t−1)
α
## (t
## −1)
α
## +1
## +r
t
i
(I,a),ifR
t−1
i
(I,a)>0
## R
t−1
i
(I,a)
## (t−1)
β
## (t
## −1)
β
## +1
## +r
t
i
(I,a),otherwise,
## C
t
i
(I,a) =C
t−1
i
(I,a)
## 
t−1
t
## 
γ
## +π
σ
t
i
(I)σ
t
i
(I,a).
LinearCFRis   a   special   case   of   DCFR,   with   up-
datesR
t
i
(I,a)   =R
t−1
i
(I,a)  +tr
t
i
(I,a),C
t
i
(I,a)   =
## C
t−1
i
(I,a)  +tπ
σ
t
i
(I)σ
t
i
(I,a).DCFR+(Hang   et   al.
2022,  2024)  integrates  CFR+  and  DCFR:R
t
i
(I,a)   =
max
## 
## R
t−1
i
(I,a)
## (t−1)
α
## (t−1)
α
## +1
## +r
t
i
(I,a),0
## 
.PCFR+(Farina,
Kroer, and Sandholm 2021) follows CFR+ for updating cu-
mulative  counterfactual  regrets  and  predicts  the  next  iter-
ation’s  cumulative  counterfactual  regrets  as
e
## R
t+1
i
(I,a)  =
max(R
t
i
(I,a) +er
t+1
i
(I,a),0), where the prediction of in-
stantaneous counterfactual regrets ̃r
t+1
i
(I,a)is assumed to
change  slowly  and  is  set  tor
t
i
(I,a).  Then  PCFR+  uses
## ̃
## R
t+1
i
(I,a)in  regret-matching  to  compute  the  next  strat-
egy.PDCFR+(Hang et al. 2024) updates cumulative coun-
terfactual  regrets  like  DCFR+  and  predicts  the  next  itera-
tion’s  cumulative  counterfactual  regrets  as
e
## R
t+1
i
(I,a)  =
max(R
t
i
(I,a)
t
α
t
α
## +1
## +er
t+1
i
(I,a),0).  For  the  cumulative
strategy,  DCFR+,  PCFR+,  and  PDCFR+  follow  the  same
formula as DCFR but differ in their choices ofγ.
Monte Carlo CFR
The tabular CFR variants improve convergence but require
full game tree traversals and tabular storage, which are in-
feasible in large games. To reduce time complexity,Monte
Carlo  CFR(MCCFR)  (Lanctot  et  al.  2009)  estimates  in-
stantaneous counterfactual regrets by sampling portions of
the game tree. MCCFR includesexternal sampling(ES),
which explores all actions for one player while sampling ac-
tions for others, andoutcome sampling(OS), which sam-
ples actions for all players along a single episode. This work
focuses on the model-free variant OS-MCCFR, which learns
directly  from  sampled  episodes  without  a  perfect  simula-
tor. At iterationt, episodes are sampled using asampling
strategyξ
t
withξ
t
i
(I,a) =ε/|A(I)|+ (1−ε)σ
t
i
(I,a)and
ξ
t
## −i
(I,a) =σ
t
## −i
(I,a). Thesampled counterfactual value
## ˆv
σ
t
i
(I,a|z)uses importance sampling to remain unbiased:
## ˆ
v
σ
t
i
## (
## I,a|z) =
π
σ
t
## −i
(z[I])π
σ
t
(z[I]a,z)u
i
## (z)
π
ξ
t
## (
z)
## =
π
σ
t
## (
z[I]a,z)u
i
## (z)
π
ξ
t
i
## (
z[
## I])π
ξ
t
(z[I],z)
## .
Thesampledinstantaneous  counterfactual  regretsis
## ˆr
t
i
(I,a) = ˆv
σ
t
i
(I,a|z)−ˆv
σ
t
i
(I|z), whereˆv
σ
t
i
(I|z) =
## P
a∈A(I)
σ
t
i
(I,a)ˆv
σ
t
i
(I,a|z). They are unbiased estima-
tors ofr
t
i
(I,a). Similarly, thesampled strategyˆσ
t
i
(I,a|z)
is defined asσ
t
i
(I,a)π
σ
t
i
(I)/π
ξ
t
## (z).
DeepCFR
DeepCFR(Brown  et  al.  2019)  uses  neural  networks  to
approximate  CFR,  avoiding  tabular  storage  of  cumulative
counterfactual regrets and cumulative strategies. In each iter-
ation, it performsKpartial traversals, stores the sampled in-
stantaneous counterfactual regrets in a reservoir buffer, and
trains a network from scratch to predict cumulative counter-
factual regrets. The expectation value of actionain an in-
formation setIafterTiterations isR
## T
i
(I,a)/
## P
## T
t=1
π
ξ
t
## (I),
where the denominator accounts for sampling bias canceled
out  during  regret  matching.  A  second  buffer  stores  strate-
gies  across  iterations  to  approximate  the  average  strategy.
For  the  sampling  strategy,  DeepCFR  adopts  ES  for  better
performance but relies on a perfect simulator, while remain-
ing compatible with OS. Since LinearCFR’s update can be
rewritten asR
t
i
(I,a) =
## P
## T
t=1
tr
t
i
(I,a), DeepCFR can ap-
proximate LinearCFR by weighing samples in iterationtby
t, yielding improved performance.
## Related Work
Learning an NE in IIGs by combining deep reinforcement
learning and game theory algorithms has gained consider-
able attention in recent years. These approaches can gener-
ally be divided into three main categories. (1) Policy-Space
## 17286

Algorithm 1:Training procedures for VR-DeepDCFR+ and VR-DeepPDCFR+
Input:total iterationsT, traversal timesK, parametersα,γ, exploration coefficientε.
Initialize each player’s cumulative advantage networkR(I,a|θ
## 0
i
)with parametersθ
## 0
i
## ;
Initialize each player’s instantaneous advantage networkr(I,a|φ
## 0
i
)with parametersφ
## 0
i
## ;
Initialize history value networkQ(h,a|w
## 0
)and history value bufferB
## Q
## ;
Initialize reservoir-sampled strategy bufferB
## Π
and each player’s advantage bufferB
## V,i
## ;
forCFR iterationt= 1toTdo
clear each player’s advantage bufferB
## V,i
## ;
foreachplayerido
fortraversalk= 1toKdo
Traverse(∅,i,B
## V,i
## ,B
## Π
## ,B
## Q
,t),▷(Algorithm 2);
## Trainθ
t
i
on lossL(θ
t
i
## ) =E
(I, ̄r)∼B
## V,i
## 
## P
a∈A(I)
## 
max(R(I,a|θ
t−1
i
## ),0)
## (t−1)
α
## (t−1)
α
## +1
+  ̄r(I,a)−R(I,a|θ
t
## )
## 
## 2
## 
## ;
For VR-DeepPDCFR+, Trainφ
t
i
on lossL(φ
t
i
## ) =E
(I, ̄r)∼B
## V,i
h
## P
a∈A(I)
( ̄r(I,a)−r(I,a|φ
t
i
## ))
## 2
i
## ;
## Trainw
t
on lossL(ω
t
## ) =E
## (t,h,a,ˆu,h
## ′
## ,I
## ′
,i)∼B
## Q
## 
## 
## ˆu+
## P
a
## ′
∈A(h
## ′
## )
σ
t+1
i
## (I
## ′
## ,a
## ′
)Q(h
## ′
## ,a
## ′
## |ω
## ′
)−Q(h,a|ω
t
## )
## 
## 2
## 
## ;
Trainψon lossL(ψ) =E
(I,t,σ
t
## )∼B
## Π
h
##  
t
## T
## 
γ
## P
a∈A(I)
## (σ
t
(I,a)−Π(I,a|ψ))
## 2
i
## ;
Output:The average strategy networkΠ(I,a|ψ).
Response Oracle (PSRO) (Lanctot et al. 2017) and its vari-
ants maintain a population of strategies and iteratively com-
pute the best response to a meta-strategy. Neural Fictitious
Self-Play  (NFSP)  (Heinrich  and  Silver  2016)  is  a  special
case of PSRO, where the meta-strategy is the uniform dis-
tribution over past strategies. While these methods are effec-
tive and scalable, they rely on computationally expensive ap-
proximate best response calculations, and their convergence
speed is often related to the size of the strategy space. (2)
Numerous neural CFR methods have been developed to ap-
proximate the behavior of tabular CFR (Brown et al. 2019;
Li et al. 2020; Gruslys et al. 2020; Steinberger, Lerer, and
Brown 2020; Liu, Li, and Togelius 2022; Meng et al. 2023;
McAleer et al. 2023). Our approach builds on this line of
work by approximating novel tabular CFR variants to further
accelerate  convergence.  (3)  Another  research  direction  in-
volves modifying policy gradient algorithms to enable con-
vergence to an NE (Srinivasan et al. 2018; Lockhart et al.
2019; Hennes et al. 2020; Fu et al. 2022). However, their
performance is sensitive to hyperparameters.
Deep (Predictive) Discounted CFR
Fitting Cumulative Advantages by Bootstrapping
DeepCFR can naturally integrate with LinearCFR but strug-
gles to approximate the behaviors of more advanced CFR
variants  like  DCFR+  and  PDCFR+,  which  rely  on  cumu-
lative  counterfactual  regrets  from  the  previous  iteration.
A straightforward approach involves approximating instan-
taneous  counterfactual  regrets  at  each  iteration  and  boot-
strapping on the estimated cumulative counterfactual regrets
from  the  prior  iteration.  However,  counterfactual  values
are  expected  utilities  weighted  by  opponents’  reach  prob-
abilities,  which  diminish  significantly  over  long  episodes.
These  reach  probabilities  vary  widely  across  information
sets, making it challenging for networks to effectively learn
values across diverse orders of magnitude (Van Hasselt et al.
2016).  Furthermore,  as  demonstrated  in  Theorem  1  (with
all  proofs  provided  in  Appendix  A),  the  expected  estima-
tion are scaled by sampling reach probabilities. This results
in the expectation of the estimated cumulative counterfac-
tual regrets taking the form
## P
## T
t=1
h
r
t
i
(I,a)/π
ξ
t
## (I)
i
## . Since
denominators change across iterations, it leads to deviation
from CFR’s behavior.
Theorem 1By  using  outcome  sampling  to  collect  data
(I,ˆr
t
i
(I))into  a  bufferB
i
for  playeriin  iterationt,  and
training  a  neural  networkr(I,a|φ
t
i
)on  lossL(φ
t
i
## )  =
## E
(I,ˆr
t
i
## (I))∼B
i
h
## P
a∈A(I)
## (ˆr
t
i
(I,a)−r(I,a|φ
t
i
## ))
## 2
i
,  the  ex-
pected target value ofr(I,a|φ
t
i
)for any sampled infor-
mation setIis given by:
## E
z∼ξ
t
## 
## ˆr
t
i
(I,a)|z∈Z
## I
## 
## =
r
t
i
(I,a)
π
ξ
t
## (I)
## .
To  address  these  challenges,  we  adjust  the  calculation
of  the  sampled  counterfactual  value  asˇv
σ
t
i
(I,a|z)  =
π
σ
t
(z[I]a,z)u
i
## (z)
π
ξ
t
(z[I],z)
,  andˇv
σ
t
i
(I|z)andˇr
t
i
(I,a)are  defined
similarly.  As  demonstrated  in  Theorem  2,  these  expecta-
tions correspond to the advantages of the information setI.
Advancements in deep reinforcement learning have shown
that  neural  networks  excel  at  predicting  and  generalizing
advantages, even in complex environments with large state
spaces  (Schulman  et  al.  2017).  By  computing  the  cumu-
lative  advantages
## ˇ
## R
t
(I,a)  =
## P
t
k=1
## A
σ
t
(I,a)instead  of
cumulative  counterfactual  regrets,  the  process  can  be  in-
terpreted  as  a  type  of  weighted  CFR  sincer
t
i
(I,a)   =
π
σ
t
## −i
## (I)A
σ
t
i
(I,a)(Srinivasan et al. 2018).
## 17287

Algorithm 2:CFR Traversal with Outcome Sampling for VR-DeepDCFR+ and VR-DeepPDCFR+
FunctionTraverse(h,i,B
## V,i
## ,B
## Π
## ,B
## Q
## ,t):
Input:Historyh, traversing playeri, advantage bufferB
## V,i
, strategy bufferB
## Π
, history value bufferB
## Q
, iterationt.
ifhis terminalthen
returnthe utility of playeri;
For VR-DeepDCFR+, compute strategyσ
t
(I,a)fromR(I,a|θ
t−1
## P(h)
)using regret matching;
For VR-DeepPDCFR+, compute strategyσ
t
(I,a)from the predicted cumulative advantages
max
## 
R(I,a|θ
t−1
## P(h)
## ),0
## 
## (t−1)
α
## (t−1)
α
## +1
+r(I,a|φ
t−1
## P(h)
)using regret matching;
fora∈A(h)do
ξ
t
(I,a)←ε
## 1
|A(h)|
## + (1−ε)σ
t
(I,a)ifP(h) =ielseσ
t
(I,a);
## ˆa∼ξ
t
(I),h
## ′
## ←hˆa;
whileP(h
## ′
)is chancedo
a∼σ(h
## ′
## ),h
## ′
## ←h
## ′
a;
̄v(I
## ′
|z)←Traverse(h
## ′
,i,B
## V,i
## ,B
## Π
## ,B
## Q
## ,t);
fora∈A(h)do
̄v(I,a|z)←Q
i
## (h,a|w
t−1
## ) +
̄v(I
## ′
|z)−Q
i
## (h,a|w
t−1
## )
ξ
t
(I,a)
ifa= ˆaelseQ
i
## (h,a|w
t−1
## )
̄v(I|z)←
## P
a∈A(h)
σ
t
(I,a) ̄v(I,a|z);
ifP(h) =ithen
fora∈A(h)do
̄r(I,a)← ̄v(I,a)−
## P
a
## ′
## ∈A(I)
σ
t
(I,a
## ′
) ̄v(I,a
## ′
## );
Insert(I, ̄r)into the advantage bufferB
## V,i
## ;
else
Insert(I,t,σ
t
(I))into the strategy bufferB
## Π
## ;
## Insert(t,h,ˆa,ˆu,h
## ′
## ,I
## ′
,i)into the history value bufferB
## Q
## ;
return ̄v(I|z);
Theorem 2By  using  outcome  sampling  to  collect  data
(I,ˇr
t
i
(I))into  a  bufferB
i
for  playeriin  iterationt,  and
training  a  neural  networkr(I,a|φ
t
i
)on  lossL(φ
t
i
## )  =
## E
(I,ˇr
t
i
## (I))∼B
i
h
## P
a∈A(I)
## (ˇr
t
i
(I,a)−r(I,a|φ
t
i
## ))
## 2
i
,  the  ex-
pected target value ofr(I,a|φ
t
i
)for any sampled infor-
mation setIis given by:
## E
z∼ξ
t
## 
## ˇr
t
i
(I,a)|z∈Z
## I
## 
## =
r
t
i
(I,a)
π
ξ
t
## −i
## (I)
## =A
σ
t
i
(I,a).
To  address  the  high  variance  introduced  by  the  impor-
tance  sampling  termπ
σ
t
i
(I)/π
ξ
t
(z)in  the  sampled  strat-
egyˆσ
t
i
(I,a|z),   which   complicates   network   train-
ing,  we  directly  use  the  strategyσ
t
i
(I,a)as  the  sam-
pled  strategyˇσ
t
i
(I,a|z).  However,  when  it  is  player
i’s   turn   to   collect   data,   we   save   the   sampled   strat-
egyˇσ
t
## −i
(I,a|z)to   the   buffer   for   the   opponent
player−i.  The  expectation  of  the  cumulative  strategy
for  player−iis  given  by
## P
## T
t=1
## E
z∈ξ
t
## 
## ˇσ
t
## −i
(I,a|z)
## 
## =
## P
## T
t=1
π
ξ
t
i
(I)π
σ
t
## −i
(I)σ
t
## −i
(I,a). This can be interpreted as a
form of weighted cumulative strategy, whereπ
ξ
t
i
(I)acts as
the weight in iterationt.
We now describe the training procedures for the cumula-
tive advantage and average strategy network. Similar to OS-
DeepCFR, outcome sampling is used to sampleKepisodes
per  iteration,  and  these  experiences  are  added  into  replay
buffers. For playeri, the replay bufferB
## V,i
stores informa-
tion setsIand advantage estimatesˇr(I,a), which are used
to train a cumulative advantage networkR(I,a|θ
t
i
)to ap-
proximate cumulative advantages at a given information set.
Unlike OS-DeepCFR, where the replay buffer retains sam-
ples  from  all  iterations,  the  replay  buffer  is  cleared  at  the
start of each iteration. The networkR(I,a|θ
t
i
)is trained by
bootstrapping according to the loss
## L
## (
θ
t
i
## ) =E
(I,ˇr)∼B
## V,i
h
## P
a∈A(I)
##  
R(I,a|θ
t−1
i
) + ˇr(I,a)−R(I,a|θ
t
i
## )
## 
## 2
i
## .
Another bufferB
## Π
with reservoir sampling stores informa-
tion setsI, iteration numberst, and sampling strategiesˇσ
t
## . It
is used to train an average networkΠ(I,a|ψ)that approx-
imates the average strategy over all iterations. The network
Π(I,a|ψ)is optimized using the following loss:
L(ψ) =E
(I,t,ˇσ
t
## )∼B
## Π
## 
## 
## X
a∈A(I)
##  
## ˇσ
t
(I,a)−Π(I,a|ψ)
## 
## 2
## 
## 
## .
Approximating Advanced CFR Variants
The estimated cumulative advantages pave the way for ap-
proximating the behaviors of DCFR+ and PDCFR+. Both
methods update cumulative counterfactual regrets by apply-
ing discounting and clipping operations. This results in the
## 17288

loss for the networkR(I,a|θ
t
i
## ):
## L
## (
θ
t
i
## ) =E
(I,ˇr)∼B
## V,i
## 
## P
a∈A(I)
## 
max(R(I,a|θ
t−1
i
## ),0)
## (t−1)
α
## (
t
## −1)
α
## +1
+ ˇr(I,a)−R(I,a|θ
t
i
## )
## 
## 2
## 
## ,
where thesequence of discounting and clipping is adjusted
to  facilitate  sampling-based  approximation  of  the  expec-
tationˇr(I,a).  Since  PDCFR+  relies  on  predicting  the  in-
stantaneous  counterfactual  regrets  for  the  next  iteration  to
compute  a  new  strategy,  an  instantaneous  advantage  net-
workr(I,a|φ
t
i
)is  trained  for  each  playeri.  This  net-
work estimates the instantaneous advantages in iterationt
using  samples  from  the  replay  bufferB
## V,i
,  with  the  loss
## L(φ
t
i
## )  =E
(I,ˇr)∼B
## V,i
h
## P
a∈A(I)
(ˇr(I,a)−r(I,a|φ
t
i
## ))
## 2
i
## .
The  instantaneous  advantage  network  is  then  used  to  pre-
dict  the  cumulative  advantages  for  the  next  iteration  as
max (R(I,a|θ
t
i
## ),0)
t
α
t
α
## +1
+r(I,a|φ
t
i
)For  the  aver-
age  strategy,  the  cumulative  strategy  can  be  expressed  as
## C
t
i
(I,a) =C
t−1
i
(I,a) +t
γ
π
σ
t
i
(I)σ
t
i
(I,a). So we can train
the average strategy network with the loss:
## L(
ψ) =E
(I,t,σ
t
## )∼B
## Π
h
##  
t
## T
## 
γ
## P
a∈A
## (I)
## (σ
t
(I,a)−Π(I,a|ψ))
## 2
i
## ,
whereTisthe total number of iterations.
Variance Reduction Based on Baseline Functions
To mitigate the high variance caused by episode sampling,
prior works such as DREAM (Steinberger, Lerer, and Brown
2020) and ESCHER (McAleer et al. 2023) incorporate value
functions at history nodes. DREAM uses value functions as
baseline functions for each action, and constructing an un-
biased estimator of counterfactual regrets, while ESCHER
directly computes counterfactual regrets using value func-
tions. Since ESCHER requires accurate value estimation and
thus incurs significant training time, we adopt the variance
reduction approach of DREAM.
For each episodezin iterationt, we extract and store a set
of experience tuples(t,h,ˆa,ˆu,h
## ′
## ,I
## ′
,i)in the history value
bufferB
## Q
.  Each  tuple  represents  playeritaking  actionˆa
at history nodeh, transitioning to nodeh
## ′
associated with
information setI
## ′
, and player 1 receiving utilityˆu(where
## ˆu=u
## 1
## (h
## ′
## )ifh
## ′
is a terminal history, andˆu= 0otherwise).
The networkQ(h,a|w
t
)estimates the value of each action
for player 1 at every history node under the strategyσ
t+1
## . It
is trained with the loss
## L
## (
ω
t
## ) =E
## (t,h,ˆa,ˆu,h
## ′
## ,I
## ′
,i)∼B
## Q
## 
## 
## ˆu+
## P
a
## ′
∈A(h
## ′
## )
σ
t+1
i
## (I
## ′
## ,a
## ′
)Q(h
## ′
## ,a
## ′
## |ω
## ′
)−Q(h,ˆa|ω
t
## )
## 
## 2
## 
## .
The network is trained in an off-policy manner, eliminating
the need to sample new episodes underσ
t+1
, thereby im-
proving  learning  efficiency.  Since  we  assume  the  game  is
two-player zero-sum, we haveQ
## 1
## (h,a|w
t
) =Q(h,a|w)
andQ
## 2
## (h,a|w
t
) =−Q(h,a|w). It is used to compute
baseline-adjusted sampled value
## ̄
v
σ
t
i
## (
## I,a|z) =
## (
## Q
i
## (h,a|w
t−1
## ) +
## ̄v
σ
t
i
## (I
## ′
|z)−Q
i
## (h,a|w
t−1
## )
ξ
t
## (
## I
## ,a)
ifa= ˆa
## Q
i
## (h,a|w
t−1
## )otherwise,
## ̄
v
σ
t
i
## (
## I|z) =
## 
u
i
## (z)ifh=z
## P
a∈A(h)
σ
t
i
(I,a) ̄v
σ
t
i
(I,a|z)otherwise.
Wethen  replace  the  sampled  advantagesˇr
t
i
(I,a|z)
with baseline-adjusted sampled advantages ̄r
t
i
(I,a|z)  =
## ̄v
σ
t
i
(I,a|z)− ̄v
σ
t
i
(I|z), which serve as unbiased estima-
tors (Schmid et al. 2019). Algorithm 1 outlines the complete
training procedure of our algorithms.
## Experiments
In   this   section,   we   evaluate   the   performance   of   VR-
DeepDCFR+ and VR-DeepPDCFR+ through extensive ex-
periments. We first demonstrate their empirical convergence
toward the NE across eight widely used IIGs in the research
community. We provide detailed descriptions of the games
in  Appendix  B.  Exploitability  is  used  as  the  performance
metric to showcase the convergence speeds. We then con-
duct  experiments  on  a  large  poker  game.  Given  the  large
size of the game, we assess performance by playing against
five agents with different styles, using average  rewards as
the performance metric. All testing games are sourced from
OpenSpiel (Lanctot et al. 2020).
We  compare  our  methods  against  five  model-free  neu-
ral  methods:  NFSP,  q-based  policy  gradient  (QPG)  /  re-
gret  policy  gradient  (RPG)  (Srinivasan  et  al.  2018),  OS-
DeepCFR and DREAM. All methods use similar network
architectures with three hidden layers with 64 neurons each.
We  normalize  the  utilities  received  by  all  algorithms  in
each  game  to  the  range[−1,1].  The  implementation  of
NFSP, QPG, and RPG are sourced from OpenSpiel, while
OS-DeepCFR is adapted from ES-DeepCFR in OpenSpiel.
Hyperparameters  for  NFSP  and  OS-DeepCFR  follow  the
OpenSpiel  reproduction  report  (Walton  and  Lisy  2021),
while those for QPG/RPG are from  (Farina and Sandholm
2021). For DREAM, we adopt the base hyperparameters of
OS-DeepCFR  and  use  a  circular  buffer  of  size  1,000,000
for  the  history  value  network.  VR-DeepDCFR+  and  VR-
DeepPDCFR+  share  same  common  hyperparameters  with
DREAM,  with  specific  settings  ofα=2,γ=2for  DeepD-
CFR+  andα=2.3,γ=2for  VR-DeepPDCFR+.  All  algo-
rithms use the same hyperparameters across all games. De-
tailed configurations are provided in Appendix C.
Convergence to Equilibrium
We  run  each  algorithm  four  times  with  different  random
seeds,  and  the  results  are  shown  in  Figure  1.  In  all  plots,
the  x-axis  is  the  number  of  episodes  sampled  by  each  al-
gorithm,  and  the  y-axis  is  exploitability  shown  on  a  log
scale. The shaded area represents95%confidence intervals
over four random seeds. The two policy gradient algorithms
QPG and RPG converge to an exploitability of 0.01 inKuhn
Poker, but perform poorly in more complex games, consis-
tent with findings in the work (Farina and Sandholm 2021).
Neural CFR variants generally converge faster than NFSP,
mainly due to CFR’s theoretically superior convergence rate.
Compared to OS-DeepCFR, the algorithms proposed in this
work converge faster in most games, demonstrating the ef-
fectiveness  of  approximating  cumulative  advantages.  Cu-
mulative  advantages  exhibit  less  variance  than  cumulative
counterfactual  regrets,  leading  to  more  stable  neural  net-
work training and faster convergence of the average strategy.
## 17289

## 0246810
## 10
## −3
## 10
## −2
## 10
## −1
## 10
## 0
## Exploitability
## Kuhn Poker
## 0246810
## 10
## −1
## 10
## 0
## Leduc Poker
## 0246810
## 10
## −1
## 10
## 0
## Battleship (2)
## 0246810
## 10
## 0
## Battleship (3)
## 0246810
## Episodes(×10
## 6
## )
## 10
## −1
## 10
## 0
## Exploitability
GoofspielImp (5)
## 0246810
## Episodes(×10
## 6
## )
## 10
## −1
## 10
## 0
GoofspielImp (6)
## 0246810
## Episodes(×10
## 6
## )
## 10
## −1
## 10
## 0
## Liar’s Dice (5)
## 0246810
## Episodes(×10
## 6
## )
## 10
## −1
## 10
## 0
## Liar’s Dice (6)
NFSPQPGRPGOS-DeepCFRDREAMVR-DeepDCFR+VR-DeepPDCFR+
Figure 1: Convergence results of seven model-free neural algorithms on eight testing games.
Moreover, VR-DeepDCFR+ and VR-DeepPDCFR+ inherit
the convergence advantages of DCFR+ and PDCFR+ over
vanilla CFR and LinearCFR. Experimental results show that
VR-DeepDCFR+  and  VR-DeepPDCFR+  outperform  OS-
DeepCFR and DREAM in convergence speed across most
games,  highlighting  the  benefit  of  integrating  neural  net-
works  with  advanced  CFR  variants.  The  running  time  of
VR-DeepDCFR+ is roughly the same as that of DREAM,
since  the  main  difference  lies  in  their  loss  formulations,
which incur negligible overhead. Moreover, by using boot-
strapping instead of retraining from scratch, it actually re-
quires fewer total training steps.
Head-to-Head Evaluation
We  evaluate  the  algorithms  on  the  large  poker  game  flop
hold’em  poker  (FHP)  by  playing  20,000  matches  against
five rule-based agents with different styles. Each agent es-
timates  hand  strength  at  decision  points  and  follows  pre-
defined rules reflecting different degrees of aggressiveness,
tightness, or bluffing. These agents simulate diverse exploit
scenarios, allowing a multidimensional assessment of strat-
egy robustness (Li and Miikkulainen 2018). Please refer to
Appendix  D  for  details.  We  use  the  average  reward  over
these matches as the performance metric.
Given the poor performance of NFSP, QPG, and RPG in
typical IIGs, we focus on comparing four neural CFR vari-
ants. We increase the number of sampled episodes to10
## 8
## ,
buffer size to10
## 7
, neurons per layer to 128, while keeping
other hyperparameters unchanged. The results are shown in
Figure  2.  Final  average  rewards  are−7.8±1.4chips  for
OS-DeepCFR,−2.0±3.1for DREAM,11.6±1.2for VR-
DeepDCFR+, and11.3±0.9for VR-DeepPDCFR+. Among
the four methods, VR-DeepDCFR+ and VR-DeepPDCFR+
consistently   outperform   various   rule-based   agents   with
different  styles.  Notably,  in  professional  Texas  Hold’em
matches, an average reward of five chips per hand is con-
sidered a significant skill gap (Morav
## ˇ
c
## ́
ık et al. 2017). There-
fore, compared to other neural CFR variants, the proposed
methods demonstrate higher learning efficiency and superior
performance in the large poker game.
## 0246810
## Episodes(×10
## 7
## )
## −20
## 0
## 20
## Average Rewards
## FHP
OS-DeepCFR
## DREAM
VR-DeepDCFR+
VR-DeepPDCFR+
Figure  2:  Head-to-head  evaluation  results  of  four  neural
CFR variants onFHP.
## Ablation Studies
The proposed algorithms consist of three key components:
bootstrapped  cumulative  advantages  estimation,  approxi-
mating advanced CFR variants, and baseline-based variance
reduction.  To  evaluate  the  impact  of  each  component,  we
use VR-DeepPDCFR+ as the base method and test perfor-
mance on four IIGs after removing each module individu-
ally.  Experimental  results  show  that  all  three  components
contribute to improved learning efficiency and overall per-
formance. Please refer to Appendix E for details.
Conclusions and Future Research
This work proposes two novel model-free neural CFR vari-
ants,  VR-DeepDCFR+  and  VR-DeepPDCFR+,  for  learn-
ing an NE in two-player zero-sum IIGs. In each iteration,
the algorithms collect variance-reduced sampled advantages
using  a  history  value  network,  bootstrap  cumulative  ad-
vantages,  and  apply  discounting  and  clipping  to  simulate
the  behaviors  of  advanced  tabular  CFR  variants  DCFR+
and  PDCFR+.  Experimental  results  demonstrate  that  our
methods  achieve  faster  convergence  across  eight  widely
used IIGs and obtain higher average rewards against vari-
ous  rule-based  agents  in  a  large  poker  game  compared  to
other  model-free  neural  algorithms.  Several  promising  di-
rections  remain  for  future  work.  One  potential  avenue  is
to  improve  the  prediction  of  instantaneous  advantages  in
VR-DeepPDCFR+,  possibly  by  leveraging  recurrent  neu-
ral networks to capture temporal dependencies more effec-
tively (Sychrovsk
## `
y et al. 2024).
## 17290

## Acknowledgments
This work is supported in part by the National Key R&D
Program of China (No. 2025ZD0122000), the Natural Sci-
ence Foundation of China (Nos. 62222606 and 61902402),
the  Key  Research  and  Development  Program  of  Jiangsu
Province (No. BE2023016), and the CCF-Baidu Open Fund.
## References
Bowling, M.; Burch, N.; Johanson, M.; and Tammelin, O.
-   Heads-up  limit  hold’em  poker  is  solved.Science,
## 347(6218): 145–149.
Brown,  N.;  Lerer,  A.;  Gross,  S.;  and  Sandholm,  T.  2019.
Deep counterfactual regret minimization.   InInternational
Conference on Machine Learning, 793–802.
Brown,  N.;  and  Sandholm,  T.  2018.   Superhuman  AI  for
heads-up  no-limit  poker:  Libratus  beats  top  professionals.
## Science, 359(6374): 418–424.
Brown,  N.;  and  Sandholm,  T.  2019a.   Solving  imperfect-
information games via discounted regret minimization.   In
AAAI Conference on Artificial Intelligence, 1829–1836.
Brown, N.; and Sandholm, T. 2019b.   Superhuman AI for
multiplayer poker.Science, 365(6456): 885–890.
Farina, G.; Kroer, C.; and Sandholm, T. 2021.  Faster game
solving via predictive Blackwell approachability: Connect-
ing regret matching and mirror descent. InAAAI Conference
on Artificial Intelligence, 5363–5371.
Farina, G.; and Sandholm, T. 2021. Model-free online learn-
ing  in  unknown  sequential  decision  making  problems  and
games. InAAAI Conference on Artificial Intelligence, 5381–
## 5390.
## Fu, H.; Liu, W.; Wu, S.; Wang, Y.; Yang, T.; Li, K.; Xing,
J.; Li, B.; Ma, B.; Fu, Q.; et al. 2022.   Actor-critic policy
optimization in a large-scale imperfect-information game. In
International Conference on Learning Representations, 1–
## 28.
Gratch, J.; Nazari, Z.; and Johnson, E. 2016.   The misrep-
resentation game: How to win at negotiation while seeming
like a nice guy. InInternational Conference on Autonomous
Agents and Multiagent Systems, 728–737.
## Gruslys, A.; Lanctot, M.; Munos, R.; Timbers, F.; Schmid,
## M.;  Perolat,  J.;  Morrill,  D.;  Zambaldi,  V.;  Lespiau,  J.-
B.;  Schultz,  J.;  Azar,  M.  G.;  Bowling,  M.;  and  Tuyls,
K.  2020.The  Advantage  Regret-Matching  Actor-Critic.
arXiv:2008.12234.
## Hang, X.; Kai, L.; Bingyun, L.; Haobo, F.; Qiang, F.; Jun-
liang, X.; and Cheng, J. 2024.  Minimizing Weighted Coun-
terfactual  Regret  with  Optimistic  Online  Mirror  Descent.
InInternational Joint Conference on Artificial Intelligence,
## 5272–5280.
Hang,  X.;  Kai,  L.;  Haobo,  F.;  Qiang,  F.;  and  Junliang,  X.
-  AutoCFR: Learning to Design Counterfactual Regret
Minimization Algorithms. InAAAI Conference on Artificial
## Intelligence, 5244–5251.
Hart, S.; and Mas-Colell, A. 2000. A simple adaptive proce-
dure leading to correlated equilibrium.Econometrica, 68(5):
## 1127–1150.
Heinrich,  J.;  and  Silver,  D.  2016.Deep  Reinforcement
Learning from Self-Play in Imperfect-Information Games.
arXiv:1603.01121.
## Hennes, D.; Morrill, D.; Omidshafiei, S.; Munos, R.; Pero-
lat, J.; Lanctot, M.; Gruslys, A.; Lespiau, J.-B.; Parmas, P.;
## Du
## ́
e
## ̃
nez-Guzm
## ́
an, E.; et al. 2020.  Neural replicator dynam-
ics:  Multiagent  learning  via  hedging  policy  gradients.   In
International Conference on Autonomous Agents and Multi-
agent Systems, 492–501.
Lanctot,  M.;  Lockhart,  E.;  Lespiau,  J.-B.;  Zambaldi,  V.;
## Upadhyay, S.; P
## ́
erolat, J.; Srinivasan, S.; Timbers, F.; Tuyls,
## K.;  Omidshafiei,  S.;  Hennes,  D.;  Morrill,  D.;  Muller,  P.;
## Ewalds, T.; Faulkner, R.; Kram
## ́
ar, J.; Vylder, B. D.; Saeta,
## B.;  Bradbury,  J.;  Ding,  D.;  Borgeaud,  S.;  Lai,  M.;  Schrit-
twieser,  J.;  Anthony,  T.;  Hughes,  E.;  Danihelka,  I.;  and
Ryan-Davis, J. 2020.   OpenSpiel: A Framework for Rein-
forcement Learning in Games. arXiv:1908.09453.
Lanctot, M.; Waugh, K.; Zinkevich, M.; and Bowling, M.
-  Monte Carlo sampling for regret minimization in ex-
tensive games.  InAdvances in Neural Information Process-
ing Systems, 1078–1086.
## Lanctot,  M.;  Zambaldi,  V.;  Gruslys,  A.;  Lazaridou,  A.;
## Tuyls, K.; P
## ́
erolat, J.; Silver, D.; and Graepel, T. 2017. A uni-
fied  game-theoretic  approach  to  multiagent  reinforcement
learning.    InAdvances  in  Neural  Information  Processing
## Systems, 4193–4206.
Li, H.; Hu, K.; Zhang, S.; Qi, Y.; and Song, L. 2020.  Dou-
ble Neural Counterfactual Regret Minimization. InInterna-
tional Conference on Learning Representations, 1–20.
Li,  X.;  and  Miikkulainen,  R.  2018.    Opponent  modeling
and exploitation in poker using evolved recurrent neural net-
works.  InGenetic and Evolutionary Computation Confer-
ence, 189–196.
Lisy, V.; Davis, T.; and Bowling, M. 2016.  Counterfactual
regret minimization in sequential security games.  InAAAI
Conference on Artificial Intelligence, 544–550.
Liu, W.; Li, B.; and Togelius, J. 2022.   Model-free neural
counterfactual regret minimization with bootstrap learning.
IEEE Transactions on Games, 15(3): 315–325.
## Lockhart, E.; Lanctot, M.; P
## ́
erolat, J.; Lespiau, J.-B.; Mor-
rill,  D.;  Timbers,  F.;  and  Tuyls,  K.  2019.   Computing  ap-
proximate equilibria in sequential adversarial games by ex-
ploitability  descent.   InInternational  Joint  Conference  on
## Artificial Intelligence, 464–470.
McAleer,  S.;  Farina,  G.;  Lanctot,  M.;  and  Sandholm,  T.
-  ESCHER: Eschewing importance sampling in games
by computing a history value function to estimate regret.  In
International Conference on Learning Representations, 1–
## 22.
Meng, L.; Ge, Z.; Tian, P.; An, B.; and Gao, Y. 2023.   An
efficient deep reinforcement learning algorithm for solving
imperfect information extensive-form games. InAAAI Con-
ference on Artificial Intelligence, 5823–5831.
## Morav
## ˇ
c
## ́
ık, M.; Schmid, M.; Burch, N.; Lis
## `
y, V.; Morrill, D.;
Bard, N.; Davis, T.; Waugh, K.; Johanson, M.; and Bowling,
M. 2017.  DeepStack: Expert-level artificial intelligence in
heads-up no-limit poker.Science, 356(6337): 508–513.
## 17291

Nash, J. J. F. 1950.  Equilibrium points in n-person games.
Proceedings  of  the  National  Academy  of  Sciences  of  the
United States of America, 36(1): 48–49.
Osborne, M. J.; and Rubinstein, A. 1994.A course in game
theory. MIT press.
Sandholm, T. 2015.  Steering evolution strategically: Com-
putational game theory and opponent exploitation for treat-
ment planning, drug design, and synthetic biology.  InAAAI
Conference on Artificial Intelligence, 4057–4061.
## Schmid, M.; Burch, N.; Lanctot, M.; Moravcik, M.; Kadlec,
R.;  and  Bowling,  M.  2019.   Variance  reduction  in  Monte
Carlo counterfactual regret minimization (VR-MCCFR) for
extensive form games using baselines.  InAAAI Conference
on Artificial Intelligence, 2157–2164.
Schulman,  J.;  Wolski,  F.;  Dhariwal,  P.;  Radford,  A.;  and
## Klimov, O. 2017. Proximal Policy Optimization Algorithms.
arXiv:1707.06347.
## Srinivasan, S.; Lanctot, M.; Zambaldi, V.; P
## ́
erolat, J.; Tuyls,
K.;  Munos,  R.;  and  Bowling,  M.  2018.   Actor-critic  pol-
icy optimization in partially observable multiagent environ-
ments.  InAdvances in Neural Information Processing Sys-
tems, 3426–3439.
Steinberger, E.; Lerer, A.; and Brown, N. 2020.  DREAM:
Deep  Regret  minimization  with  Advantage  baselines  and
Model-free learning. arXiv:2006.10410.
## Sychrovsk
## `
y, D.;
## ˇ
## Sustr, M.; Davoodi, E.; Bowling, M.; Lanc-
tot, M.; and Schmid, M. 2024.   Learning not to regret.   In
AAAI Conference on Artificial Intelligence, 15202–15210.
## Tammelin, O. 2014.   Solving Large Imperfect Information
Games Using CFR+. arXiv:1407.5042.
Van Hasselt, H. P.; Guez, A.; Hessel, M.; Mnih, V.; and Sil-
ver, D. 2016.  Learning values across many orders of mag-
nitude.  InAdvances in Neural Information Processing Sys-
tems, 4294–4302.
Walton,  M.;  and  Lisy,  V.  2021.Multi-agent  Reinforce-
ment   Learning   in   OpenSpiel:   A   Reproduction   Report.
arXiv:2103.00187.
Zinkevich,  M.;  Johanson,  M.;  Bowling,  M.;  and  Piccione,
C.  2007.   Regret  minimization  in  games  with  incomplete
information. InAdvances in Neural Information Processing
## Systems, 1729–1736.
## 17292