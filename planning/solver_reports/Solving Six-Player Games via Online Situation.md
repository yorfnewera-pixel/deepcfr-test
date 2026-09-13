

Solving Six-Player Games via Online Situation
## Estimation
## Huale Li
## Computer Application Research Center
Harbin Institute of Technology, ShenZhen
ShenZhen, China
hualeli@cs.hitsz.edu.cn
## Xuan Wang
## Computer Application Research Center
Harbin Institute of Technology, ShenZhen
ShenZhen, China
wangxuan@cs.hitsz.edu.cn
## Shuhan Qi
## *
## Computer Application Research Center
Harbin Institute of Technology, ShenZhen
ShenZhen, China
shuhanqi@cs.hitsz.edu.cn
## Yang Liu
## Computer Application Research Center
Harbin Institute of Technology, ShenZhen
ShenZhen, China
liu.yang@hit.edu.cn
## Haojie Wang
## Computer Application Research Center
Harbin Institute of Technology, ShenZhen
ShenZhen, China
haojiewang@cs.hitsz.edu.cn
## Fengwei Jia
## Computer Application Research Center
Harbin Institute of Technology, ShenZhen
ShenZhen, China
jfw129@gmail.com
## Jiajia Zhang
## Computer Application Research Center
Harbin Institute of Technology, ShenZhen
ShenZhen, China
zhangjj@pcl.ac.cn
Abstract—While the artificial intelligence theory for solving
the perfect-information games has been well developed in recent
years, great challenges are still posed in dealing with the
imperfect-information game due to the huge state space and
hidden information involved in it. In this paper, we design
an online strategy solving framework for six-player no-limit
Texas hold’em poker. Based on hand isomorphism and hand
strength evalution, the framework provides an efficient situation
estimation method for six-player poker. Such method could
greatly reduce the the state space in six-player poker as well
as effectively evaluate the current hands. The poker agent based
on our method won the third place in the 2018 AAAI-ACPC.
Index  Terms—game, imperfect information, poker, six-player
## I.  INTRODUCTION
Computer  games  have  always  been  regarded  as  the  touch-
stone  to  verify  the  theory  of  artificial  intelligence,  and  it  is
also  one  of  the  most  attractive  research  domain  of  artificial
intelligence.  Generally  speaking,  the  computer  game  can  be
divided  into  perfect-information  game  (PIG)  and  imperfect-
information game (IIG). The PIG refers to the game in which
each player could access to the accurate information, including
characteristics,  strategy  spaces  and  profit  functions,  etc.,  of
other players. The game of Go, as a typical PIG, presents such
a situation that the positions of all chess pieces are visible to
both of the players. By contrast, certain information is hidden
in the IIG. Therefore each player holds specfic information that
cannot be overserved by other players. Since the IIG has such
hidden information, it is much more difficult and challenging
than the PIG [1].
*Corresponding author.
For  a  long  time,  Texas  hold’em  poker  has  been  seen  as
the  most  representative  game  to  research  the  IIG  [1],  [2],
since its large state space and complex strategy. For example,
heads-up  no-limit  hold’em  (HUNL,  one  type  of  the  Texas
hold’em  poker)  is  an  extremely  complex  game  with10
## 165
information sets. In the progress of playing, each player has to
make decisions without knowing all the cards of other players,
and  meanwhile,  tries  to  judge  bluffing  from  opponents.  The
research about the IIG has attracted much attention in recent
years. For example, counterfactual regret minimization (CFR)
is  a  popular  and  effective  method  for  solving  the  IIG  [2].
Bowling proposes an algorithm framework for solving approx-
imate  Nash  equilibrium  using  MC-CFR  based  on  the  state
space compression [3]. DeepStack is designed for the IIG like
poker [4], which beats professional human poker players in the
HUNL  in  44,000  hands.  Brown  et  al.  puts  forward  Libratus,
which defeates four top human players in the 120,000 hands
HUNL tournament. It solves the leading benchmark problem
and the long-standing challenge of the IIG [5], [6].
Although the IIG agents like Deepstack and Libratus have
shown high level intelligence in the field of the HUNL, such
methods only designed for the two-player IIG [4], [6]. With the
number of the player increases, the state space of the game will
be explosively increasing, which makes the strategy solving of
multi-player game become much more difficult. Despite some
methods have been proposed to solve the strategy of the three-
player poker [7], the multi-player poker is still a challenging
problem in the field of the IIG.
This paper contributes to the study of decreasing the impact
of  large  state  space  problem  and  increasing  the  efficiency
## 1795
2019 IEEE 31st International Conference on Tools with Artificial Intelligence (ICTAI)
## 2375-0197/19/$31.00 ©2019 IEEE
## DOI 10.1109/ICTAI.2019.00270
Authorized licensed use limited to: University of Wollongong. Downloaded on August 16,2020 at 07:26:14 UTC from IEEE Xplore.  Restrictions apply.

of  obtaining  the  effective  strategy.  It  does  so  by  presenting
a  strategy  solving  framework  for  six-player  no-limit  Texas
hold’em(SPNL),  which  is  able  to  solve  the  online  strategy
in a limited time. A multi-player situation estimation method
for six-player poker, which is based on the hand isomorphism
and the hand strength evalution, is introduced in SPNL. Such
method could greatly reduce the the state space in six-player
poker as well as effectively evaluate the current hands. In addi-
tion, compared with methods that utilized with neural network,
less  history  data  of  playing  is  required  in  our  methods.  The
agent  based  on  our  method  won  the  third  place  in  the  2018
AAAI-ACPC (Annual Computer Poker Competition, a poker
competition held by AAAI and IJCAI in turn) six-player no-
limit Texas holdem poker.
## II.  B
## ACKGROUND
A. Zero-Sum Games
1) Nash Equilibrium:Nash  equilibrium  strategy  or  ap-
proximate   Nash   equilibrium   strategy   are   usually   used   in
the  IIG  with  two  players.  It  is  a  guideline  about  how  to
act   on   every   information   set   that   might   arrive   [8],   [9].
The  strategyσ
i
of  the  playerifor  each  information  set
## I
i
## ∈L
i
## ,σ
i
## (I
i
## ):A(I
i
)→[0,1]is   a   probability   distribution
function   in   the   information   setA(I
i
).   A   strategy   profile
composed   of   the   strategy   of   each   player,   which   is   the
σ=(σ
## 1
## ,σ
## 2
, ..., σ
n
## ).σ
## −i
means the strategy profile composed
of the strategy of the remaining players except the playeri.
Nash  equilibrium  is  a  strategy  profileσ=(σ
## 1
## ,σ
## 2
, ..., σ
n
## )
fornplayers. When the state of the game is in the Nash equi-
librium, no participant can get a higher utility by unilaterally
changing their strategy. The mathematical representation of the
Nash equilibrium is as follows:
u
i
## (σ)≥max
σ
## ∗
i
## ∈Σ
i
u
i
## (σ
## ∗
i
## ,σ
## −i
## ),(1)
whereu
i
is the utility of the playeri.
2) Counterfactual  Regret:CFR(Counterfactual    Regret
Minimization) is a popular algorithm to solve the Nash equi-
librium strategy, which is based on the minimization of regret
values  [2].  The  core  of  the  Minimization  algorithm  of  regret
value lies in the exploration of the Nash equilibrium [8].
The CFR contains three steps: Firstly, the regret minimiza-
tion  algorithm  compares  the  utility  of  each  strategy  with  the
average  utility  and  obtain  the  difference.  Secondly,  the  algo-
rithm selects the corresponding strategy next time according to
the difference obtained before. Finally, the algorithm chooses
the  next  appropriate  strategy  in  the  game  according  to  the
difference of size. It can be regarded as a balance if average
values from both sides of players are less than the regret value.
The method of calculating the average regret value is depicted
as:
## R
## T
t
## =
## 1
## T
max
σ
## ∗
i
## ∈Σ
i
## T
## ∑
t=1
## (u
i
## (σ
## ∗
i
## ,σ
t
## −i
## )−u
i
## (σ
t
## )),(2)
where  T  is  the  number  of  games  played,  andu
i
is  the
counterfactual value.
## 
## ;
## $
## %
## 
Fig. 1.   Action mapping under Lipschitz’s continuity
In recent years, CFR algorithm and its related variants are
widely used to calculate the Nash equilibrium stragety in the
IIG [2], [3], [5]. Although CFR and its variants have achieved
good  results  in  solving  the  Nash  equilibrium,  they  are  only
solving two-player games.
## B. Game Abstraction
Due  to  the  huge  state  space  of  the  IIG,  the  state  space  of
the game need to be compressed in usual. Action abstraction
is  the  common  method  to  compress  the  state  space  for  the
no-limit poker.
1) Action Abstraction:In limit Texas hold’em, players can
take three actions: fold, call, raise, and the number of chips for
raising is fixed. However, in no-limit Texas hold’em poker, the
number of raising chips is nearly equivalent to be continuous,
which results in plenty of actions and a large breadth of game
tree  [9]–[11].  Clustering  the  betting  actions  of  similar  chips
can greatly reduce the breadth of the game tree, and the betting
chips  according  to  different  multiples  of  the  current  pot  are
clustered usually [12]–[16].
The strategy that is calculated after the action abstraction is
applied to the original game. The action taken by the opponent
has  to  be  mapped  to  the  abstract  action  set.  The  action
mapping algorithm needs to meet the theoretical requirements
of boundary constraint, monotonicity, scale invariance, robust-
ness  and  boundary  robustness.  The  algorithm  based  on  the
probability  distribution  used  in  Tartannian  5  is  a  common
action abstraction algorithm [17].
Assuming  thatS∈[T,T
## 
]is  the  legal  chip  in  a  given
information  set,  raising  chips  of  the  opponents  arex,  and
the  abstract  action  set  A,  B  satisfyA=max{A
i
## :A
i
## x}
andB=min{A
i
## :A
i
>x}. The problem of action mapping
can  be  transformed  into  the  problem  of  mappingxto  A  or
B  according  to  the  probability.  The  probability  distribution
function satisfying Lipschitz’s continuity is:
f
## A,B
## =
(B−x)(1 +A)
(B−A)(1 +x)
## ,(3)
as shown in Fig.1, the actual betting action is mapped to A or
B according to the calculated probability.
## III.  O
## URMETHOD
The state space of the HUNL is about10
## 165
. The existing
methods,  such  as  DeepStack  and  Libratus,  can  deal  with  the
HUNL  [4],  [6].  However,  due  to  the  huge  state  space  of
the  SPNL  which  is  much  larger  than  that  of  the  HUNL,
the  existing  method  is  not  suitable  for  this  kind  of  games.
## 1796
Authorized licensed use limited to: University of Wollongong. Downloaded on August 16,2020 at 07:26:14 UTC from IEEE Xplore.  Restrictions apply.

## 2ULJLQDO*DPHIRU631/
## $EVWUDFW*DPH
## +DQG,VRPRUSKLVP
## IRU631/
## 6LWXDWLRQ(VWLPDWLRQ
## )RU631/
## %HWWLQJ6WUDWHJ\
## 2QOLQH
## 6WUDWHJ\
## 6ROYLQJ
## )UDPHZRUN
## IRU631/
## DFWLRQ
## DFWLRQ
## DFWLRQ1
## 
Fig. 2.   The online strategy solving framework for SPNL
According  to  the  characteristics  of  the  six-player  game,  we
propose  a  situation  estimation  method  for  six-player  poker
based on the hand isomorphism and the hand strength evalu-
tion. There are three steps in our method. Firstly, the original
game  is  abstracted  into  an  abstract  game  with  smaller  state
space, which maintains main properties of the original game.
Secondly, the multi-player situation estimation method is used
to evalutae the current state. Finally, the strategy of the abstract
game  is  obtained  by  the  betting  strategy.  The  online  strategy
solving framework for SPNL is shown in Fig.2.
A. Hand Isomorphism for SPNL
In Texas hold’em, some hands will lead to the same strategy
because there is no difference between different colors. In the
pre-flop, for example, hA-s10, sA-d10 and dA-h10 should be
considered  as  the  same.  That  is,  the  corresponding  solving
strategy  of  these  hands  should  be  the  same.  Therefore,  the
size of the game tree can be greatly reduced by clustering the
same type of hand combinations into the same kind. We refer
to  the  hand  isomorphism  algorithm  in  [17],  and  design  the
algorithm  specialized  for  the  SPNL.  Note  that  the  values  of
the  index  of  the  hand  with  the  same  color  are  the  same  by
indexing the hand of the player.
The algorithm that used to reduce the size of the state space
is depicted in algorithm 1.
In  the  algorithm,  the  hand  isomorphism  algorithm  indexs
each  hand.  The  index  of  the  hand  of  pattern  isomorphisms
consists of four rounds: pre-flop, flop, turn and river. In each
round,  new  cards  are  issued.  Three  steps  are  taken  to  index
the hand:
Step 1. Index the hand in each round.
Step  2.  Establish  the  value  of  the  index  of  multi-round’s
hands based on the results of the index of the previous step.
Step  3.  Introduce  the  concept  of  color  configuration  and
establish the final value of the index.
B. Situation Estimation for SPNL
1) Current hand strength:The  hand  of  the  player  is  the
combination   of   private   cards   and   public   cards.   Thus   the
strength  of  the  hand  can  be  expressed  as  a  probability.  For
example, Texas hold’em poker has 52 cards, and the hands of
opponents can be represented by the remaining 50 cards in the
pre-flop round. In this case, our hand is better(+1), tied(0) and
worse(-1) compared with the hand strength of opponents. The
hand strength can be computed through taking the summation
and divided by the total number of possible opponents’ hands.
It is depicted as:
## HS=
## N
## Better
## −N
## Worse
## N
## ,(4)
whereNis the number of possible poker situations. Compared
with the opponents’ hand strength,N
## Better
is the number of
better cases andN
## Worse
is the number of worse cases.
Algorithm 1The hand isomorphism algorithm for SPNL
Input: HandC
## 1
## ,C
## 2
## , ... ,C
k
Output: Index of hands
1: Map a hand to typical hand according to its pattern and
color,C
## 
## 1
## ,C
## 
## 2
## , ... ,C
## 
k
2:foriin{c, d, s, t}do
3:Index the same color of theKwheel, A
## 1
## ;A
## 2
## , ... , A
k
## 4:B←A
## 1
## 5:next←indexgroup
m2,...,mk
## (A
## 2
## , ... , A
k
## |∪A
## 1
## )
## 6:idx←
## (
## N−|u|
## |A
## 1
## |
## )
next
7:forifrom 1to|A
## 1
## |do
8:b←largest(B)
9:rank←b-|smaller(b)∩V|
## 10:idx←idx+
## (
rank
## |A
i
## |+i−1
## )
11:Delete b from set B
12:end for
13:Idx←Idx+
## (
## N−M+1
## M
## )
14:end for
15:returnIndex
Supposing  our  hand  is  hA-rJ  in  the  pre-flop,  the  opponent
has  1225  kinds  of  cards  (two  out  of  50  cards,  assuming  one
opponent).  In  the  flop,  the  public  card  is  rQ-h6-d5,  then  the
opponent’s  hand  has  1081  kinds  of  cards.  In  this  case,  our
hand strength is 0.585, that is to say, when our hand is hA-rJ,
the chance of our hand better than opponent’s hand is 58.5%.
2) Current hand potential:The strength of the combination
of private cards and public cards are considered together in the
most of times while calculating the current hand strength HS.
Considering that the current hand strength does not distinguish
the private cards from the public cards, and hence the player
do  not  know  whether  the  private  cards,  the  public  cards  or
both of them have an impact on the current hand strength HS.
The current hand strength may be the combination of smaller
private  cards  and  larger  public  cards,  or  the  combination  of
larger private cards and smaller public cards, while the HS is
invariant.  Moreover,  there  are  two  public  cards  after  the  flop
round, which will also change the hand strength HS.
Therefore, it is necessary to calculate the hand potential of
the  current  hand.  The  hand  potential  refers  to  the  possibility
## 1797
Authorized licensed use limited to: University of Wollongong. Downloaded on August 16,2020 at 07:26:14 UTC from IEEE Xplore.  Restrictions apply.

that the combination of the following public cards and private
cards  can  improve  the  winning  probability  of  the  previous
hand.  The  players  have  to  identify  the  potential  impact  of
public  cards  later.  With  public  cards  are  issued,  the  positive
potential  will  increase  the  probability  of  winning  and  vice
versa.
3) Effective hand strength:Effective  hand  strength  com-
bines current hand strength and current hand potential, which
gives an estimation of the winning probability of current hand
strength compared with the opponents.
## P
## Pot
is the number of times when the current hand strength
is behind the opponents but ends up ahead the opponents.N
## Pot
is the number of times when the current hand strength is ahead
the opponents but ends up behind the opponents.
Pr(win)=HS×(1−N
## Pot
## )+(1−HS)×P
## Pot
## (5)
In many cases, when one has the best hands at the moment
without  considering  the  potential  negative  effects,  the  choice
of betting is a good action. Hence, what players are interested
in  is  that  whether  their  hands  are  the  best  at  the  moment  or
the probability to become the best will be improved, which is
measured by the hand strength. In the SPNL, the EHS is used
to estimate the current hand strength, and the formula for the
effective hand strength EHS is as follows:
## EHS=HS+(1−HS)×P
## Pot
## .(6)
C. Betting Strategy for SPNL
In  limit  Texas  hold’em  poker,  players  can  only  take  three
actions: fold, call, and raise, the number of chips for raising is
fixed. All the possible actions can be taken into account when
solving Nash equilibrium since the branch of the game tree is
small  in  the  sequence  of  actions.  In  no-limit  Texas  hold’em,
however,  players  always  have  a  wide  range  of  options.  For
example, compared with heads-up limit Texas hold’em, whose
state space of is10
## 17
, the state space of HUNL is10
## 165
## . The
reason  for  the  large  space  of  no-limit  Texas  hold’em  is  that
there  are  too  many  actions  that  the  players  can  choose.  The
state space of SPNL is much larger than that of HUNL, since
the number of the player in SPNL increasing.
There  are  many  available  actions  for  players  in  no-limit
Texas hold’em, however, professional players usually take the
action  of  raising  multiples  of  the  current  pot.  For  example,
when  the  current  pot  is  150,  the  number  of  chips  raised  by
the  player  is  500.  In  this  case,  the  action  of  raising  500  is
considered as a huge betting action. If the current pot is 1500,
the same action of raising 500 is considered to be a medium
betting  action.  The  same  chips  are  regarded  as  different  bets
in different situations. To make an efficient decision in such a
huge  action  space,  a  heuristic  approach  is  proposed  to  make
the betting action strategy. The number of chips in the paper
is multiple of the current pot.
The  basic  betting  strategy  for  the  SPNL  is  the  following
four steps:
Step  1.  Carry  out  the  hand  isomorphism  and  establish  the
value of the index.
Step  2.  Calculate  the  EHS  of  the  current  hand  relative  to
the hand of the other five opponents.
Step 3. Transform EHS into the probability of the action by
using  a  set  of  betting  rules  and  formulas  :  Pr(fold),  Pr(call),
## Pr(raise).
Step  4.  Generate  random  numbers  and  use  these  numbers
to select actions from the probability distribution.
## IV.  E
## XPERIMENTS
ACPC  (Annual  Computer  Poker  Competition)  is  a  poker
competition  held  by  AAAI  and  IJCAI  in  turn.  In  this  paper,
we  design  a  betting  strategy  for  SPNL,  and  use  ACPC  as
an  experiment  platform.  SPNL  is  one  of  the  ACPC-2018
competitions  and  the  poker  agent  based  on  our  method  got
the third place.
A. ACPC Communication
At the beginning of the game, the poker agent establishes a
connection with the server through the specified port number
and communicates with the server using TCP/IP protocol. In
games,  poker  agents  do  not  communicate  directly,  but  are
forwarded through the server side. The communication format
between the poker agent and the server is specified as follows:
< serverMessage >:=< matchState >
< matchState >:=MATCHSTATE:< position ><
handNumber >< betting >< cards >
The string of communication is composed of 5 parts, each
part is separated by colon. The meaning of each character is
shown in Table I.
## TABLE I
## T
## HE MEANING OF THE CHARACTER
CharacterMeaning
MatchstateIdentifying this information is normol cammunication
between the agent and the server.
PositionRepresenting the player’s position, 0 indicates a big
blind, and 1 indicates a small blind.
HandNumberIndicates what inning is in the game
bettingRepresents the action sequences of the players, c (call),
f (fold), r (raise)
cardsRepresents public cards and private cards. d, c, s and h
indicate colors.
## B. Experiment Setup
We conduct experiments on SPNL, which is a standard and
new benchmark in 2018 ACPC. The players start each game
with  20,000  chips  and  alternate  positions  after  each  game.
Each  player  can  choose  to  either  fold,  call,  or  raise  on  each
of the four rounds of betting. Calling means the player places
a  number  of  chips  in  the  pot  that  is  equal  to  the  opponents’
share.  Raising  means  the  player  adds  more  chips  to  the  pot
than  the  opponents’  share.  Players  cannot  raise  beyond  the
20,000  they  start  with.  At  the  start  of  each  game  of  SPNL,
both  players  are  dealt  two  private  cards  from  a  standard  52-
card deck. Details of rules about SPNL in our experiment are
depicted in Table   II.
## 1798
Authorized licensed use limited to: University of Wollongong. Downloaded on August 16,2020 at 07:26:14 UTC from IEEE Xplore.  Restrictions apply.

## C. Experiment Result
Our experiment is to play with the poker agents in the ACPC
SPNL. The agents come from the top six in 2018 ACPC SPNL
competition.  Because  the  participants  of  these  related  agents
did not provide relevant open source code, nor did they publish
relevant implementation details. Thus, the result we provide is
the official comparison result of 2018 ACPC. Our agent runs
with RAM of 256G and 64 CPU cores. In the experiments, the
results  are  measured  in  the  standard  units  used  in  this  field:
milli big blinds per hand (mbb/h).
## TABLE II
## E
## XPERIMENTSDETAILS
## Hands Per Match3000
Stack Sizes200 big blinds (400 small blinds)
Bet SizesNo limit
## Blind Sizes50/100
Blind StructureReverse blinds, no ascending blinds
Illegal ActionsAny illegal action is interpreted as a call
Time per hand2 seconds
## TABLE III
## E
## XPERIMENTRESULTS IN2018 ACPC
AgentResults(mbb/h)
our agent1298 +/- 189
PokerBot53336 +/- 265
## Paco1669 +/- 159
## Slumbot910 +/- 167
PhantomX-2389 +/- 268
## Feste-4824 +/- 243
In the Table   III, the result is composed of two values. The
first  value  is  the  expected  winnings,  in  thousandths  of  a  big
blind  per  hand,  for  the  row  player.  The  second  value  is  the
95%confidence interval around the mean.
## TABLE IV
## O
## PPONENT AGENTS AND THEIR METHODS
AgentMethod
PokerBot5online opponent modeling, Bayesian graphical modeling
PacoMC self-play, neural networks, a hand-crafted policy
SlumbotTargeted CFR [18]
PhantomXpre-computed strategy generated through MCTS
FesteCFR [2]
Table  IV shows the the opponent agents and their methods
in our experiments. To our knowledge, in these poker agents,
PokerBot5 uses an online opponent modeling approach where
the opponent strategies are modeled using Bayesian graphical
modeling.  Paco  attempts  to  approximate  a  Nash  equilibrium
using  a  combination  of  Monte  Carlo  self-play  and  neural
networks,  as  well  as  a  hand-crafted  policy.  Slumbot  uses
Targeted CFR to approximate a Nash equilibrium. PhantomX
uses a pre-computed strategy generated through Monte Carlo
tree search. Feste attempts to approximate a Nash equilibrium
using CFR.
## V.   C
## ONCLUSION
This  paper  deals  with  solving  the  strategy  of  six-player
no-limit  Texas  hold’em.  We  design  a  framework  to  solve
the  online  strategy  of  six-player  no-limit  Texas  hold’em.  A
situation  estimation  method  is  proposed  to  reduce  the  state
space  in  the  six-player  poker  and  to  efficiently  evaluate  the
current  hand  strength.  Our  method  offers  a  new  idea  to
solve the SPNL, and meanwhile, compared with methods that
utilized  with  neural  network,  less  history  data  of  playing  is
required in our methods.
## A
## CKNOWLEDGMENT
This  research  is  supported  by  Key  Technology  Program
ofShenzhen,China,(No.JSGG20170823152809704),
KeyTechnologyProgramofShenzhen,China,
(No.JSGG20170824163239586),    and    BasicResearchProject
of Shenzhen, China, (No.JCYJ20180507183624136).
## R
## EFERENCES
[1]   Billings, Darse and Davidson, Aaron and Schaeffer, Jonathan and Szafron,
Duane,  ”The  challenge  of  poker,”  Artificial  Intelligence,  vol.  134,  pp.
## 201–240, 2002.
[2]   Zinkevich,  Martin  and  Johanson,  Michael  and  Bowling,  Michael  and
Piccione,  Carmelo,  ”Regret  minimization  in  games  with  incomplete
information,” NIPS, vol. 134, pp. 1729–1736, 2008.
[3]   Bowling, Michael and Burch, Neil and Johanson, Michael and Tammelin,
Oskari, ”Heads-up limit hold’em poker is solved,” Science, vol. 347, pp.
## 145–149, 2015.
## [4]   Morav
## ˇ
c
## ́
ık, Matej and Schmid, Martin and Burch, Neil and Lis
## `
y, Viliam
and Morrill, Dustin and Bard, Nolan and Davis, Trevor and Waugh, Kevin
and Johanson, Michael and Bowling, Michael, ”Deepstack: Expert-level
artificial  intelligence  in  heads-up  no-limit  poker,”  Artificial  Intelligence,
vol. 1356, pp. 508–513, 2017.
[5]   Brown, Noam and Sandholm, Tuomas, ”Safe and nested subgame solving
for imperfect-information games,” NIPS, pp. 689–699, 2017.
[6]   Brown,  Noam  and  Sandholm,  Tuomas,  ”Superhuman  AI  for  heads-up
no-limit  poker:  Libratus  beats  top  professionals,”  Science,  vol.  359,  pp.
## 418–424, 2018.
[7]   Gibson, Richard G, ”Regret minimization in games and the development
of champion multiplayer computer poker-playing agents,” 2014.
[8]   Nash, John, ”Non-cooperative games,” Annals of mathematics, pp. 286–
## 295, 1951.
[9]   Billings, Darse and Burch, Neil and Davidson, Aaron and Holte, Robert
and  Schaeffer,  Jonathan  and  Schauenberg,  Terence  and  Szafron,  Duane,
”Approximating  game-theoretic  optimal  strategies  for  full-scale  poker,”
IJCAI, Mexico, vol. 3, p. 661, 2003.
[10]   Johanson, Michael, ”Measuring the size of large no-limit poker games,”
## Computer Science, 2013.
[11]   Zhang, Jiajia and Hong, Liu, ”Reinforcement Learning with Monte Carlo
Sampling in Imperfect Information Problems,” ICCC, China, 2018.
[12]   Johanson, Michael and Burch, Neil and Valenzano, Richard and Bowl-
ing,   Michael,   ”Evaluating   state-space   abstractions   in   extensive-form
games,” AAMAS, Italy, pp. 271–278, 2013.
[13]   Sandholm, Tuomas, ”The state of solving large incomplete-information
games, and application to poker,” AI Magazine, vol. 31, pp. 13–32, 2010.
[14]   Shi,  Jiefu  and  Littman,  Michael  L,  ”Abstraction  methods  for  game
theoretic poker,” CGAMES, pp. 333–345, 2000.
[15]   Jia, Fengwei and Wang, Xuan and Guan, Jian and Qi, Shuhan and Liao,
Qing and Li, Huale, ”Bi-directional Features Reuse Network for Salient
Object Detection,” PRICAI, Fiji, 2019.
[16]   Zhang,  Jiajia  and  Hong,  Liu,  ”Building  Endgame  Data  set  to  Improve
Opponent Modeling Approach,” DSC, China, 2017.
[17]   Waugh,  Kevin,  ”A  fast  and  optimal  hand  isomorphism  algorithm,”
Workshops at AAAI, USA, 2013.
[18]   Jackson, Eric Griffin, ”Targeted CFR,” Workshops at AAAI, USA, 2017.
## 1799
Authorized licensed use limited to: University of Wollongong. Downloaded on August 16,2020 at 07:26:14 UTC from IEEE Xplore.  Restrictions apply.