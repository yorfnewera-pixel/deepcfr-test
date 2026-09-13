

Knowledge-Based Systems 231 (2021) 107434
Contents lists available at ScienceDirect
Knowledge-BasedSystems
journal homepage: www.elsevier.com/locate/knosys
## Scalablesub-gamesolvingforimperfect-informationgames
HualeLi
a
,XuanWang
a,b
,KunchiLi
a
,FengweiJia
a
,YulinWu
a
,JiajiaZhang
a,
## ∗
## ,
ShuhanQi
a,b,
## ∗∗
a
School of Computer Science and Technology, Harbin Institute of Technology, ShenZhen, 518055, China
b
Peng cheng Laboratory, 518038, Shenzhen, China
a r t i c l e    i n f o
Article history:
## Received 31 August 2020
Received in revised form 7 April 2021
## Accepted 21 August 2021
Available online 26 August 2021
## Keywords:
## Game
## Counterfactualregretminimization
## Imperfect-information
## Agent
a b s t r a c t
Counterfactual regret minimization (CFR) is a popular and effective method for solving a game with
imperfect information. The effect of CFR is limited by the size of the game state space. With the
increase in the number of game participants, the game state space will increase rapidly. Although
the vanilla CFR is suitable for two-player imperfect-information games, it does not work well in
imperfect-information games with three or more players. In this paper, we design a framework for
imperfect-information games, which can not only deal with two-player imperfect-information games
but also can efficiently solve three-player imperfect-information games. Compared with traditional
solving methods, in this framework we propose real-time hand abstraction (RTHA), which can reduce
theerrorcausedbytheabstraction.Wealsoproposeawarm-startonlinesolutionofsub-game(WSOS-
SG) method, which can improve the accuracy of the action estimation and solve the sub-game in real
time. Experimental results show that the agent based on our method achieve better performances
than traditional methods. The agent based on our method took part in the 2018 AAAI-ACPC poker
competition and won third place in heads-up no-limit Texas hold’em.
©2021ElsevierB.V.Allrightsreserved.
## 1. Introduction
In recent decades, games have been an important testbed
for studying how well artificial intelligence (AI) can perform in
complicated decision making. In games with perfect information,
players usually can find either an optimal action or an optimal
distributionofactionsforthecurrentstate.InthegameofGo,for
example, all the information is public for each player, and some
search algorithms can be used to seek an optimal action in the
currentstateofthegame.BycombiningaMonteCarlotreesearch
(MCTS) and deep reinforcement learning, AlphaGo defeated a
Go world champion [1–5]. The later improved versions, AlphaGo
Zero and Alpha Zero, drew lessons from AlphaGo, which allowed
computers to surpass human beings completely in the game
of Go [6,7]. However, in many real world applications, such as
auctions,negotiations,andstocktrades,theinformationisalways
imperfect for each game player, which limits the application of
perfect-information game methods.
In imperfect-information games, each player aims to find a
Nash equilibrium or an approximate Nash equilibrium such that
## ∗
Corresponding author.
## ∗∗
Corresponding author at: School of Computer Science and Technology,
Harbin Institute of Technology, ShenZhen, 518055, China.
E-mail addresses:zhangjiajia@hit.edu.cn (J. Zhang),
shuhanqi@cs.hitsz.edu.cn (S. Qi).
no single player can improve their utility by deviating from the
equilibrium [8]. Currently, to achieve such a goal, counterfactual
regret minimization (CFR) and its variants are the most popular
algorithms for seeking a Nash equilibrium or an approximate
Nash equilibrium in games with imperfect information [9]. By
combining the regret minimization algorithm and average strat-
egy calculation method and using an extensive game model to
represent the problem of sequential decision making, CFR itera-
tively converges to a Nash equilibrium or an approximate Nash
equilibrium in two-player zero-sum games [10].
In recent years, many improvements based on the vanilla
CFR have been proposed. Johanson et al. extended the sampling
nodes in the procedure of searching the game tree and proposed
threeCFRalgorithmvariantsbasedonchancenodesampling[11].
Lanctot et al. designed a general sampling framework for the
vanilla CFR and proposed a class of online algorithms based on
a Monte Carlo approach (MCCFR), which included CFR based on
outcomesampling(OS)andexternalsampling(ES)[12].TheMC-
CFR accelerates the speed of the overall convergence by reducing
the cost of a single iteration and increasing the number of itera-
tions. As a result, the scale of the problem solved by the MCCFR
is about 10 times larger than that of the vanilla CFR. Burch et al.
proposedCFR-decomposition(CFR-D)thatcouldhandlethetrunk
and each sub-problem independently [13,14]. Bowling et al. im-
proved the regret value matching algorithm in CFR and proposed
the CFR+ algorithm with a faster convergence speed [15]. Brown
https://doi.org/10.1016/j.knosys.2021.107434
0950-7051/©2021 Elsevier B.V. All rights reserved.

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
et al. sped up the convergence of the CFR and proposed a CFR
algorithm based on regret value pruning [16]. Jackson proposed
a CFR-based method, Compact CFR, for imperfect-information
games [17]. Further, Jackson proposed a new sampling variant
of the CFR, Targeted CFR [18], which can be viewed as falling
somewhere between outcome sampling and external sampling
on the spectrum of sampling algorithms. Brown et al. proposed
DeepCFR[19],whichcombinedaneuralnetworkandvanillaCFR.
Texas hold’em poker is a game with imperfect information,
and it is a major event of the ACPC (Annual Computer Poker
Competition, held by AAAI and IJCAI in turn) [20,21]. The agents
HITSZ_HKL[22],Gibson_3p[23] and the agent based on neural
fictitious self-play (NFSP) [24] are the agents with excellent per-
formance in the ACPC in these years. As a major event of the
ACPC, Heads-up no-limit hold’em (HUNL) is also one of the most
popular games, which contains 10
## 161
states. In the HUNL, each
player has private information, and the state space grows as the
number of players is increased. However, the vanilla CFR can
only solve games with 10
## 18
states. The traditional method to
solve large-scale games with imperfect information is to abstract
the original game before using CFR [25,26]. By using such meth-
ods, Deepstack, Libratus, and Pluribus have successfully defeated
professional human players [27–30].
## Manyachievementshavebeenmadeinthefieldofimperfect-
information games over the past few years, such as the de-
velopment of Deepstack and Libratus. However, most of these
methods focus on games with two players (such as HUNL) [27,
29] and cannot be used in multi-player games (such as three-
player no-limit poker), which are extremely common in real-life
scenarios [31]. Although Pluribus has recently defeated human
professional players in six-player Texas poker [30], determining
howtosolvethemulti-playerimperfect-informationgameisstill
a challenge problem.
## Inthepaper,wedealwiththisproblembydesigningaframe-
work to solve imperfect-information games. The framework con-
sists of real-time hand abstraction (RTHA) and a warm-start on-
line solution of sub-game (WSOS-SG) method. The RTHA is used
to abstract player hands in real time during the process of game
abstraction.TheWSOS-SGcansolvethesub-gamestrategyonline
in real time during the game. The key idea of the WSOS-SG
is that we introduce a warm-start CFR-based method, which
can increase the understanding of the situation information and
improve the accuracy of the action estimation. Moreover, in the
WSOS-SG, we propose a state value estimation method, which
increasestheestimationprecisionoftheintermediatestatevalue
of the sub-game tree.
Noting that our method has strong flexibility, and it can
be adapted to two-player to three-player imperfect-information
games. In experiments with HUNL and three-player no-limit
Texas hold’em poker (TPNL), we showed that the agent based
on the proposed method performed much better than some
high-level agents. Our agent based on the proposed method
participatedinthe2018-ACPCandfinishedinthethirdplace.We
summarize our contributions as follows:
(1) We designed a framework based on abstraction and CFR,
which was used to study the solutions of imperfect-information
games with two and three players.
(2) In the designed framework, RTHA and WSOS-SG based on
the CFR are presented to solve the strategy of the imperfect-
information game, which reduce the error caused by abstraction
and improve the accuracy of the final solution strategy.
(3) Experimental results showed that the agent based on our
method achieved better performances compared to traditional
methods.
The rest of our paper is organized as follows. In Section 2,
we introduce the concepts of the extensive-form game, describe
the Nash equilibrium and CFR algorithm, and discuss the strictly
dominant strategy. Our method is presented in Section 3, the
frameworkisintroducedinSection3.1,andtheRTHAinthegame
abstractionisintroducedinSection3.2.Furthermore,thestrategy
solvingmethodWSOS-SGisdiscussedinSection3.3.InSection4,
we evaluate the performance of our method in THUL and TPNL.
Finally, in Section 5, the content of the paper is summarized, and
future work is discussed.
## 2. Background
We first introduce the notation and definitions of extensive-
form games. We then introduce the details of the Nash equilib-
rium.Afterthis,wediscussthepopularCFRalgorithm.Finally,we
conclude this section by discussing the concepts of the strictly
dominant strategy. Since many variables will be used in this
paper, a detailed description of the variables is given in Table 1.
2.1. Extensive-form game
## Theextensive-formgameisaclassicalmodelforanimperfect-
information game. A game tree can be used to represent the
extensive-form game. The unique initial state of the game is rep-
resented by the root of the game tree. Choice nodes are possible
decision points of each player. The edges leaving a choice node
representthelegalactionsthatcanbetakenbythecorresponding
player.Terminalnodesrepresenttheendofthegameandcontain
theutilitiesofallplayersreachingthatpoint[32,33].Fig.1shows
a game tree of the game rock–paper–scissors. As shown in Fig. 1,
player 1 takes the action ‘rock,’ player 2 also takes the action
‘rock,’andthentheutilityofbotharezero.Thisisthefullprocess
of the game.
There are often several states that a player cannot differen-
tiate, since opponents have private information (such as their
private cards in poker) in imperfect-information games. Here
informationsetsarethesesetsofallindistinguishablegamestates
for that player. Each choice node corresponds to an information
set (instead of including a node for every game state) in an
extensive-form game with imperfect information. In poker, there
are many more game states than information sets.
Definition 1.A finite extensive-form gameΓwith imperfect
information has the following six components⟨N,H,P,f
c
,u,I⟩:
(1) A finite setNof players,N=1,2,...,n;
(2)AfinitesetHofsequencesofpossibleactionhistories.The
empty sequence and every prefix of a sequence are also inH.
Z⊆Hare the terminal histories, which are not a prefix of any
other sequences.A(h)=
## {
a|(h,a)∈H
## }
are the actions available
after a non-terminal historyh∈H.
(3) A functionPthat assigns to each non-terminal history
(each member ofH\Z) a member ofN∪
## {
c
## }
, wherePis the
player function.P(h) is the player who will take an action after
thehistoryh.IfP(h)=c,thenchancedeterminestheactionafter
the historyh.
(4) A functionf
c
associated with every historyhfor which
P(h)=cis a probability measuref
c
(·|h) onA(h) (f
c
(·|h) is the
probability thataoccurs givenh), where each such probability
measure is independent of every other such measure.
(5) For each playeri⊆H, a partitionI
i
of
## {
h∈H:P(h)=i
## }
with the property thatA(h)=A(h
## ′
) wheneverhandh
## ′
are in the
same member of the partition.
(6) For each playeri⊆N, a utility functionu
i
from the
terminal statesZto the realsR. IfN=1,2 andu
## 1
## =u
## 2
, it is
a zero-sum extensive game.
## 2

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
## Table 1
Detailed descriptions of the variables used in the paper.
VariableDetailed description
HA finite set of sequences, possible histories of actions.his a history (i.e., node),h∈H.Z⊆Hare the terminal histories.
IThe information set in imperfect-information games. For any information setI
i
belonging to playeri, all nodesh,h
## ′
## ∈I
i
are indistinguishable to the playeri.
A(h)The actions available at a nodeh.
p(h)The player whose turn it is to act at the nodeh. If chance acts the node, thenP(h)=c.
u
i
A utility functionu
i
from the terminal statesZto the realsR. IfN=1,2; andu
## 1
## =u
## 2
, it is a zero-sum game.
σ
i
A probability vector over actions for the playeriin the information setI. A strategy profileσis a tuple of strategies,
one for each player.σ
## −i
is the strategy of all players other than the playeri.
π
σ
(h)The probability with whichhis reached if all players play according to the strategyσ.π
σ
i
(h) is the contribution ofito
this probability.π
σ
## −i
(h) is the contribution of chance and all players other thani.
## R
## T
i
(I,a)Average overall regret in the CFR, defined in Eq. (4).
tThe number of the iteration in the CFR,Tis the total iteration number.
v
i
(σ,I)The counterfactual value of an information setIwhereP(I)=iis the expected utility to playerigiven thatIhas been
reached, weighed by the external reach ofIfor playeri, as shown in Eq. (5).
Fig. 1.Game tree of the game rock–paper–scissors (circles represent the players of the game, boxes represent the benefits after the game, and the connecting lines
between the circles represent the legal actions allowed by the game.).
Definition 2.A strategyσ
i
of playeriin an extensive-form game
isafunctionthatassignsadistributionoverA(I
i
)toeachI
i
,andΣ
i
isthesetofstrategiesfortheplayeri.Astrategyprofileσconsists
of a strategy for each player,σ
## 1
## ,σ
## 2
## ,...,σ
n
, whereσ
## −i
refers to
all the strategies inσexceptσ
i
## .
## Letπ
σ
(h) be the probability of historyhoccurring if players
chooseactionsaccordingtoσ.Wecandecomposeπ
σ
i
## (h)
## ∏
i∈N∪
## {
c
## }
π
σ
i
(h) into each player’s contribution to this probability. Hence,
π
σ
i
(h) is the probability that if the playeriplays according toσ,
thenforallhistoriesh
## ′
thatareaproperprefixofhwithP(h
## ′
## )=i,
the playeritakes the corresponding action inh. Letπ
σ
## −i
(h) be the
productofallplayers’contributions(includingchance)exceptthe
playeri. ForI⊆H, defineπ
σ
## (I)=Σ
h∈I
π
σ
(h), as the probability
of reaching a particular information set givenσ, withπ
σ
i
(I) and
π
σ
## −i
(I) defined similarly.
The overall value to the playeriof a strategy profile is the
expectedpayoffoftheresultingterminalnode,u
i
(σ)=Σ
h∈Z
u
i
## (h)
π
σ
## (h).
## 2.2. Nash Equilibrium
Nash equilibrium andε-Nash equilibrium are often used in
two-player zero-sum games [8]. A Nash equilibrium is a strategy
profileσ, in which no player can increase their utility by unilat-
erally changing their strategy. That is, if each player is provided
with the static strategies of all the other players and still cannot
gain any utility by changing their strategy, then they are all
playing at a Nash equilibrium. We introduce the best response
to present Nash equilibrium more clearly.
Definition 3.A best responseb
i
is a strategy that obtains the
highest utility against the set of all other strategies in a strategy
profileσ:
b
i
## (σ
## −i
## )=max
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
## ),(1)
whereΣ
i
represents all possible strategy profiles for the players
andu
i
represents the benefits they obtain.
In poker, this corresponds to the strategy that wins the most
chips against all other players, given that their entire strategies
havebeenrevealedandremainstatic(theydonotchange).Abest
responsetoanabstractgameisconfinedtoplayingintheabstract
game (as opposed to the original game).
Notice that each player in a Nash equilibrium is a best re-
sponse to the set of all other players. More formally, a strategy
profile fornplayers,σ
## 1
## ,σ
## 2
## ,...,σ
n
, is a Nash equilibrium if it
satisfies the following constraints:
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
## ),(2)
whereσ
## ∗
i
isaNashequilibriumfortheplayeri.Thus,weseethat,
in a Nash equilibrium, there is no strategy for playeriinΣ
i
that
yieldsahigherutilityagainstσ
## −i
thanitsstrategy,σ
i
,inσ.Infact,
in a two-player zero-sum game, each player who plays a Nash
equilibriumstrategyhasthesamevalueregardlessofwhichNash
equilibrium strategy each player plays.
Computing an exact Nash equilibrium for a large game such
as poker (or even an abstraction) is infeasible. Thus, we are
ofteninterestedinfindingapproximationstoNashequilibria,the
ε-Nash equilibrium, such that
u
i
## (σ)+ε≥max
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
## ).(3)
Anε-Nash equilibrium is a strategy profile,σ, in which no
player can increase their utility more thanεby unilaterally
changing their strategy.
## 3

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
2.3. Counterfactual regret minimization (CFR)
CFRhasbeenapopularalgorithmforfindingNashequilibriain
largeimperfect-informationgamesoverthepastseveralyears[9].
CFR is an iterative algorithm that minimizes a form of regret,
positive immediate counterfactual regret, for each information
set. As the number of iterations increases, the average overall
regret decreases, and the iteration strategy will approach a Nash
equilibrium. Average overall regret is defined as follows:
## R
## T
i
(I,a)=
## 1
## T
## T
## ∑
t=1
## (v
i
(I,σ
t
## I→a
## )−v
i
(I,σ
t
## )),(4)
whereTisthenumberofgamesplayed,andσ
t
## I→a
isanequivalent
strategy profile ofσ, except that playerichooses the actiona
in the information setIinσ
t
## I→a
## .v
i
is the counterfactual value,
defined as follows:
v
i
(σ,I)=
## ∑
z∈Z
## I
u
i
## (z)π
σ
## −i
(z[I])π
σ
(z[I],z),(5)
whereZ
## I
is the action sequence of the information setI,z[I]is
the action sequence from the root node to the information setI,
u
i
(z) is the utility of playeriobtained in the leaf nodez,π
σ
## −i
(z[I])
is the probability that the opponent reaches the information set
I, andπ
σ
(z[I],Z) is the product of the probability for all players
from the information setIto the leaf nodez.
The next iteration strategy in the information setIcan be
represented as follows:
σ
## T+1
i
(I,a)=
## ⎧
## ⎨
## ⎩
## R
## T,+
i
(T,a)
## Σ
a∈A(I)
## R
## T,+
i
(T,a)
## ,Σ
a∈A(I)
## R
## T,+
i
(T,a)>0
## 1
## |
## A(I)
## |
## ,Σ
a∈A(I)
## R
## T,+
i
(T,a)≤0
## (6)
The average strategy of the players computed with the CFR
will eventually converge to Nash equilibrium through a large
number of iterations, and the average strategy is as follows:
## ̄σ
## T
i
(I,a)=
## Σ
## T
t=1
π
σ
t
i
(I)σ
t
(I,a)
## Σ
## T
t=1
π
σ
t
i
## (I)
## .(7)
Since the CFR was proposed, many variants have appeared
with improved computational speeds, game tree sizes, and sam-
pling methods, such as CFR+, Linear CFR, MCCFR and CFR-D [11–
## 13,15,34,35].
2.4. Strictly dominant strategy
Since the CFR has no theoretical guarantee for multiplayer
games, the strictly dominant strategy is introduced to solve the
sub-optimal strategy in multiplayer games by using the CFR. The
CFRcanremoveastrictlydominantstrategyafteralargenumber
of strategy iterations, and Gibson [36] verified the feasibility of
theCFRinsolvingmulti-playergameswithimperfectinformation
through experiments. The definition of the dominant action and
the dominant strategy are as follows:
Definition 4.The action of playeriisa∈A(I) in an extensive-
form game. If exists another strategyσ
## ′
i
## ∈Σ
i
for playeri, any
strategyprofileσ
## −i
## ∈Σ
## −i
forotherplayerstherearev
i
(I,σ
## I→a
## )≤
v
i
(I,(σ
## ′
i
## ,σ
## −i
)),theactionaisaweakdominantaction.Inthiscase,
the dominant action can be defined recursively as an action that
is dominant or becomes dominant after eliminating the strictly
dominant action in the process of the strategy iteration.
Definition 5.Astrategyσ
i
ofplayeriisaweakdominantstrategy
if there exists another strategyσ
## ′
i
that satisfies the following
conditions:
(1)Foranystrategyσ
## −i
## ∈Σ
## −i
fortheotherplayers,u
i
## (σ
i
## ,σ
## −i
## )
## ≤u
i
## (σ
## ′
i
## ,σ
## −i
## );
(2) For some strategiesσ
## −i
## ∈Σ
## −i
for the other players,
u
i
## (σ
i
## ,σ
## −i
## )<u
i
## (σ
## ′
i
## ,σ
## −i
## ).
## Ifallstrategiesσ
## −i
## ∈Σ
## −i
fortheotherplayerssatisfyu
i
## (σ
i
## ,σ
## −i
## )
## <u
i
## (σ
## ′
i
## ,σ
## −i
## ),thenσ
i
isastrictlydominantstrategy.Thedominant
strategycanbedefinedrecursivelyasastrategythatisdominant
or becomes dominant after eliminating the strictly dominant
strategy in the strategy iteration process.
- Our method
## Inthissection,wepresentthedetailsofourproposedmethod.
First, we introduce the framework for solving an imperfect-
information game. We then introduce the RTHA for game ab-
straction. Finally, we discuss the details of the proposed strategy
solving method WSOS-SG.
MostofexistingmethodsthatusetheCFRtoseekaNashequi-
librium aim at solving two-player imperfect-information games.
## Inthispaper,tosolvemulti-playerimperfect-informationgames,
we design a framework to solve the game strategy. In the frame-
work, we propose the RTHA to abstract player hands in real
time. We also propose a warm-start based method for sub-game
onlinesolutionsolving(WSOS-SG),whichconsistsofthreeparts:
a real-time hand distribution prediction (RTHD), intermediate
state value estimation in sub-game (ISVE-SG) method, and a
warm-start solution of sub-game method based on CFR (WS-SG).
3.1. Framework overview
We designed a framework to solve the strategy of the
imperfect-information game. Our framework can solve the strat-
egy of imperfect-information games with two and three players
effectively. Moreover, the framework can be easily extended
to imperfect-information games with over three players. The
framework is shown in Fig. 2.
The framework of the proposed strategy solving process
mainly consists of two parts: game abstraction and strategy
solving. We propose the RTHA to abstract player hands in real
time in the game abstraction process, and the WSOS-SG process
to solve the strategy in the game, which includes RTHD, ISVE-SG,
and WS-SG.
3.2. Game abstraction
## Theimperfect-informationgamecontainsprivateinformation,
and the state space becomes much larger as the number of the
player increases. Thus, the strategy solution of an imperfect-
information game is more complicated than that of an perfect-
information game. To solve this problem, we propose the RTHA
method, which can abstract the game state in real time.
Real-Time Hand Abstraction (RTHA)Since the number of
possiblecombinationsofprivateandpubliccardsdecreasesasthe
game progresses, the state space of the game tree can be greatly
reducedbyremovingtheimpossiblecombinationsofhandsfrom
the issued public cards, and then the hands are abstracted. Ta-
ble 2 shows the number of all combinations and the number of
combinations after removing some impossible combinations in
threerounds.Thus,real-timehandabstractioncangreatlyreduce
the state space of the game and improve the accuracy of hand
abstraction as the game progresses.
The RTHA mainly includes the following two steps:
Step 1: Calculate the strength values of various combinations
of the hands abstracted by the hand isomorphism algorithm
through offline abstraction [37], which can accelerate the speed
of the calculation when using hand abstraction in real time.
## 4

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
Fig. 2.Framework of strategy solving. Since the original game has a larger state space than the abstract game, the triangle of the original game is larger than the
other two triangles.
## Table 2
Number of combinations in three rounds.
RoundNumber of original
combinations
Number of combinations
after removing public cards
## Flop280948000001271256
## Turn140470000001128
## River28094760001081
Step 2: When abstracting the hands in real time, different
kindsofhandsareabstractedbasedonthepositionofthecurrent
gamesituationintheoriginalgametree.First,ifthecurrentsitu-
ation of the game is in the pre-flop round, the offline abstraction
by hand isomorphism from step 1 is used directly. Meanwhile,
the total number of combinations of the hands is C(52,2) = 1326,
and the number of combinations of hands after using hand iso-
morphism is 169. Second, if the game is currently in the flop,
turn, or river round, the algorithm removes the public cards that
have been issued. Third, the algorithm looks up the strength of
the possible combinations of hands calculated in step 1. Finally,
the k-means algorithm [38] is used for clustering the hands.
3.3. Strategy solving with WSOS-SG
After the state space of the game is compressed by the RTHA,
the method of strategy solving of the WSOS-SG for imperfect
information is used in the process of solving the game.
3.3.1. Real-time hand distribution prediction (RTHD)
As the game progresses, players expose more and more infor-
mation to their opponents, because the actions they have taken
are related to the player’s hand strength. Here the strength of
the hand can be expressed as a probability. For example, Texas
hold’em poker has 52 cards, and the hands of opponents can be
represented by the remaining 50 cards in the preflop round. In
this case, our hand is better(+1), tied(0) and worse(-1) compared
with the hand strength of opponents. The hand strength can be
computedthroughtakingthesummationanddividedbythetotal
number of possible opponents’ hands. If a player takes the action
ofraisingchipsinthepre-flopandfloprounds,thisplayer’shand
strength may be strong; the more chips the player raises, the
greater the strength of the hand the player may have. Further-
more,iftheplayeronlytakestheactionofcalling,thestrengthof
the player’s hand may be weak. The behavior of the same player
isalwaysstable.Inviewofthis,weanalyzetheplayer’splaystyle
according to their game data history (such as betting actions). In
this way, we predict the player’s private hands more accurately
according to the player’s actions, which is very similar to human
behavior of reading hands in playing games.
The establishment of the specific model of a player is needed,
when we predict the player’s hand distribution according to the
actionsequencetakenbytheplayer.However,itcannotbeeffec-
tively established in the absence of the history of the game data
Algorithm 1Algorithm for RTHD
Input:Active playersP=P
## 1
## ,···,P
n
, public cardsB, number
of private cardsH, current betting historyh, the original hand
distributionDofactiveplayers,thetrunkstrategyσorastrategy
σ
i
of the sub-game, and the last round action sequenceAs.
Output:Normalized hand distributionsD.
1:forplayeri∈Pdo
## 2:D
## ′
i
## ←[0]
## H
3:forP
## 1
=0 to 50,P
## 1
is not onBdo
4:forP
## 2
=1 to 51,P
## 2
is not onBdo
5:I←IndexHoles(P
## 1
## ,P
## 2
); //build hand index of each
player
## 6:p
i
←the probability of the playeritakes action
a
i
∈Asaccording to the strategyσorσ
i
## ;
## 7:D
## ′
i
[I]←p
i
## ;
8:end for
9:end for
## 10:D
i
## ←D
i
## ⊙D
## ′
i
; //multiply the corresponding element
11:Normalize(D
i
); // normalize the hand distribution of each
player
12:end for
13:returnD.
foraspecificplayer.Inthispaper,weassumethattheopponent’s
strategy is the real-time strategy we calculated, and then Bayes’
formula [39] is used to calculate the player’s hand distribution
according to the action taken by the player.
Assuming thatσis the real-time strategy computed by the
agent, all the possible hands of the playeriareC
## 1
## ,C
## 2
## ,...,C
n
## .
ActionAwillbetakenaccordingtothestrategyσ.Theconditional
probability of the player’s hand after taking actionAcan be
calculated as follows:
## P(C
i
## )=P(C
i
## |A)
## =
## P(A|C
i
## )×P(C
i
## )
## ΣP(A|C
j
## )×P(C
j
## )
## ,
## (8)
whereP(A|C
j
) is the probability of taking the actionAwhen
## C
j
is the hand of the playeri. The normalization ofP(C
i
) is the
distribution of the hand after the playeritakes the actionA. The
whole process is depicted in Algorithm 1.
3.3.2. Intermediate state value estimation in sub-game (ISVE-SG)
When calculating the strategy of the current game, the size of
the sub-game tree is determined by the position of the current
situation in the original game tree. If leaf nodes of the sub-
game tree do not reach the end of the game, it is necessary
to estimate the value of these intermediate leaf nodes. There
are many methods to estimate the payoff of leaf nodes in the
sub-game tree, such as MCTS, and deep neural networks [2,40].
Since many existing value estimation methods of leaf nodes
are based on the comparison of the current players’ hand
## 5

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
Algorithm 2Algorithm for ISVE-SG
Input:Active playersN, current game stateS, the hand dis-
tribution of playersD, the action sequence historyh, the hand
abstractionC, the strategy of players in the last iterationσ, and
the value of the leaf nodev.
Output:Value of the leaf nodev.
Establish a sub-game tree with the current leaf node as the root
node.
1:Initialize:CurrentNode←S;v←[ ];
2:ifCurrentNode∈T, the game is end in this roundthen
3:Renewvby comparing the hand strengths of players;
## 4:else
5:Renewvwith UCT or MCTS;
6:end if
## 7:returnv.
strengths. There are some problems in calculating the values of
the leaf nodes through these methods. First, the exploitability of
the calculated strategy of the game is relatively high (e.g., the
agentdoesnotconsidersomeoftheopponents’skillsandismore
direct in choosing actions). Second, it needs to assume that the
opponent has adopted the strategy we calculated before, and if
thecalculatedstrategyisbad,thentheerrorofthepredictionwill
be large. As the game progresses, errors will gradually increase.
To solve the above problems, we propose ISVE-SG method. In
theISVE-SG,dependingonwhetherthesolutionofthesub-game
tree can reach the leaf nodes, we divide the estimation of the
intermediate state into two cases:
Case 1: A sub-game tree is established with the current leaf
node as the root node, and the leaf node of the sub-game tree
only reaches the end of the current round. If the leaf node of the
sub-game tree is at the end of the game, the player’s utility is
estimated by the hand strength.
Case 2: If the leaf node of the sub-game tree is not at the end
of the game, the algorithm of upper confidence bound apply to
tree (UCT) [41] or MCTS [2] is used to calculate the value of the
current leaf node.
## Throughtheabovemethods,wedealwiththedifferentarrival
modesoftheleafnodesinthesub-gamewithfinergranularityto
improve the accuracy of the intermediate state value estimation.
The ISVE-SG in our paper is shown in Algorithm 2.
3.3.3. Warm start online solution of sub-game based on CFR (WS-SG)
We proposed the WS-SG, which can not only solve the prob-
lem of a two-player game with imperfect information but can
also effectively solve the strategy of a three-player game with
imperfect information. Due to the huge state space of imperfect-
information games, the loss of information is significant by using
the game abstraction, and the scale of the state space increases
exponentially with the increase in the number of players. Thus,
it is unreasonable to consider the game as a whole to solve the
strategy.
Compared with the traditional method of solving game strat-
egy based on state abstraction, the WS-SG no longer regards the
game tree as a whole but establishes an abstract sub-game tree
through state abstraction for each current game. Our method
estimates the value at the leaf node of the sub-game tree and
then uses the CFR or MCTS algorithm to calculate the strategy of
the current game in real time. With the progress of the game,
more and more actions are taken by players, and public cards
are issued gradually. Based on the actions taken by players and
the public cards issued, the distribution of opponents’ hands is
predicted in real time. Meanwhile, the chips of the current pot
areusedastheinputoftheagenttocorrecttheestimationofthe
pot by the agent. When solving the strategy of the current game,
the distribution of each player’s hand can be predicted by using
the actions taken by each player before and the public cards are
issued.
## Bayes’formula[39]isusedtocalculatethecurrentdistribution
of each player’s hand according to the strategy computed in
the previous step, and agents can further reduce the size of the
sub-game tree through RTHA based on the issued public cards.
The proposed WS-SG method is shown in Algorithm 3. The
WS-SG mainly includes two parts: the trunk strategy (computed
offline, only used in the pre-flop round), and the online strategy.
The WS-SG is completed according to the following steps:
Step 1: Abstract the hand in real time with the hand abstrac-
tion algorithm and compute a trunk strategy;
Step2:Judgewhetheritistheturnoftheagenttotakeaction:
a. If it is, renew the distribution of the hand of the player,
establish a sub-tree of the game, compute the real-time strategy
of the current state of the game, and take action based on the
real-time strategy;
b. If it is not, renew the strategy of the players, take action
accordingtothepreviousactionsequence,renewthedistribution
of the hands of the players, and update the real-time strategies
of the players;
Step 3: The game continues until the end, and the next loop
begins.
Algorithm 3Algorithm for WS-SG
Input:ActiveplayersN,currentgamestateS,originalgametree
T, the hand distribution of playersD=D
## 1
## ,···,D
## N
, the action
sequencehistoryh,thehandabstractionC,thestrategyofplayers
in the last iterationσ, and the trunk strategyσ
t
## .
Follow the trunk strategyσ
t
until a leaf node or some mid-game
is reached.
Establish a sub-game tree with the current leaf node as the root
node.
1:Initialize:CurrentNode←S;As←[]; //traversefromcurrent
state
2:ifCurrentNode∈Tand we are activethen
3:RenewDwith trunk strategyσ
t
, and betting historyh;
4:end if
5:whileCurrentNode∈Tand the game is not enddo
6:ifCurrentNode∈beginning of betting roundthen
7:Removethecombinationwiththepubliccards,usethe
RTHA to abstract the hand, normalize the probability of the
hand distribution;
8:ifAs̸=[]then
9:RenewDwithAs,σ, and the betting historyh;
10:As←[ ];
11:end if
12:else ifCurrentNode/∈beginning of betting roundthen
13:ifAs̸=[]then
14:RenewDwithAs,σ, and the betting historyh;
15:As←[ ];
16:end if
17:Establish a sub-game tree, use CFR to update strategy
σ, and wait for the active player to take actiona;
18:S←ChildState(S,a); //assign successor node to the
current node
19:As.append(a);
20:end if
21:end while
## 6

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
## Table 3
Details of the two kinds of poker.
Game Player Round Total cards Public cards Private cards Big blind
## HUNL 2   4    5252100
## TPNL 3   4    5252100
## 4. Experiments
In this section, we quantitatively evaluate the performance
of the agents based on our method and compare with existing
agents in the HUNL and the TPNL.
4.1. Experiment setup
Toevaluatetheproposedmethod,weselectedHUNLandTPNL
poker as the research objects. The experiments were conducted
based on the rules of the ACPC, which is held by the AAAI/IJCAI
every year. The ACPC specifies that each agent’s total chip value
isupdatedto20000,thesmallblindis50chips,andthebigblind
is 100 chips when the game begins.
TheHUNLisatwo-playerpokergame.Itisarepeatedgame,in
which the two players play a match of individual games, usually
called hands, while alternating who is the dealer. In each of the
individualgames,oneplayerwillwinsomenumberofchipsfrom
theotherplayer,andthegoalistowinasmanychipsaspossible
over the course of the match. Each individual game begins with
both players placing a number of chips in the pot: the player in
the dealer position puts in the small blind, and the other player
puts in the big blind, which is twice the small blind amount.
During a game, a player can only wager and win up to a fixed
amount known as their stack, in the format of HUNL used in the
ACPC and this paper, the big blind is 100 chips and the stack is
20000 chips or 200 big blinds.
HUNL and TPNL both consist of four rounds: pre-flop, flop,
turn, and river. There are 52 cards in total. In the pre-flop round,
each player is dealt two cards by the dealer. These two cards are
theplayer’sprivatehandsandareunobservedbytheiropponents.
In the flop round, the dealer deals three public cards. A card is
dealt both in the turn and the river. Finally, there are a total of
five public cards over four rounds on the board. Each player can
formthestrongestfive-cardhandtocomparewithotherplayers.
Each player can use any cards from their two private cards and
five public cards to form their hand.
After cards for the round are dealt, players alternate taking
actions of three types: fold, call, or raise. Folding means giving
up on the current game and forfeiting all wagers placed in the
pot. Calling means the player places a number of chips in the
pot to equal to the opponent’s share. Raising means the player
adds more chips to the pot than the opponent’s share. Players
cannot raise beyond the 20000 chips they start with under the
ACPC rules.
Experiments were performed following the ACPC rules. The
agents in the experiments applied 32 threads to accelerate the
computation of the real-time strategy. The RTHA was used to
decrease the number of the game states for all the experiments.
During the pre-flop, the abstraction buckets the hand into 169
abstract hands. During the other three rounds, the abstraction
bucketsthehandinto200abstracthands.Becauseno-limitpoker
has a huge action space (the number of bet chips is continuous if
the chips are available), we adopt action abstraction [25]. In our
experiments, the game included five bet sizes for every round of
the game (0.5, 1.0, 2.0, and 5.0 times the size of the current pot
and all in). A player can also decide not to bet. Table 3 provides
more details of the two types of poker.
Evaluation criterion: The performance of the agent is mea-
sured in terms of milli big blinds per game (mbb/g), which is a
standard measure of the win rate in poker games. Researchers
have standardized on the unit milli-big-blinds per game, or
mbb/g, where one milli-big-blind is one thousandth of one big
blind.
4.2. Result comparison
Inthissection,theproposedmethodsaretestedonHUNLand
TPNL poker. To objectively test the performance of our method
and the effectiveness of each improved module, we selected two
strong agents for comparison,HITSZ_HKLandGibson_3p[22,23].
HITSZ_HKL[22] is the agent that took the third place in HUNL
in the 2017-ACPC, which uses the traditional state abstraction
method and then calculates the approximate Nash equilibrium
with the Pure-CFR algorithm to obtain the offline strategy. Card
abstractionisusedintheHITSZ_HKLagent.Theabstractionbuck-
etsthehandinto169,300,1000,and1000abstractshandsinturn
in the pre-flop, flop, turn, and river, respectively. TheHITSZ_HKL
includes five bet sizes at every round in the game (0.75, 1.0, 2.0,
and 3.0 times the size of the current pot and all in). Another
agent,Gibson_3p[23],wasappliedastheopponentagentforTPNL
poker, which does not use card abstraction. It includes three bet
sizesineveryround(0.5and1.0timesthesizeofthecurrentpot
and all in).
We conducted four groups of experiments to verify our
method. We designed four agents (agent1, agent2, agent3, and
agent4)intheexperiments.Theperformancesofouragentswere
gradually enhanced to test the effectiveness of each module.
In other words, the experimental results of our final agent are
shown in Table 7. In addition, for agent4 implemented with our
method,wealsoselectedsixhigh-levelagentsreportedinrecent
years to carry out comparative experiments. The experimental
results are shown in Table 8.
All four agents used the RTHA and RTHD to abstract the
state of the game and renew the hand distribution, respectively.
Whensolvingthestrategy,agent1usedUCTandagent2usedCFR.
Agent3usedRTHDandCFRtosolvethestrategy.Agent4usedthe
WSOS-SG to compute the strategy in the games. All four agents
could be used in both HUNL and TPNL poker. Table 4 shows the
agents and their methods.
4.2.1. Performance of the online and offline strategy
Our first experiment compared the performance of the online
and offline strategies. For strategy solving, we used the CFR
and UCT algorithms. Table 5 shows the performance of each
technique. In all our experiments, the results were measured in
the standard units used in this field: milli-big-blinds per hand
## (mbb/h).
As shown in Table 5, agent1 and agent2 based on the online
strategy were worse than the agents based on the state abstrac-
tion and the offline strategy, both losing more than 500 mbb/h
over 20000 hands. This was because the online strategy received
very little information at the beginning of the game. Meanwhile,
the error of the strategy using real-time computing directly was
relativelylarge.Moreover,inthefirstroundofpoker,publiccards
were not issued, and the scale of the game state and the value
estimation error of the intermediate state were relatively large,
which resulted in the non-ideal performance of the agent.
In addition, the experimental results in Table 5 showed that
agent1 based on the UCT was worse than agent2 based on CFR.
This was mainly because the UCT required a certain number of
searches, but the real-time calculation was limited by the time
and the computer hardware, which resulted in an insufficient
number of searches. Thus, in later experiments, we used the CFR
to solve the strategy if there was no special explanation.
## 7

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
## Table 4
Agents considered in this paper and their methods.
AgentMethod
HITSZ_HKL   Traditional state abstraction method, uses Pure-CFR to solve strategy, only for two-player games, in [22]
Gibson_3p   Uses CFR to solve strict dominated strategy, only for three-player games, in [23]
Agent1Uses RTHA to abstract the original game, uses RTHD to renew hand distribution, uses UCT to solve strategy
Agent2Uses RTHA to abstract the original game, uses RTHD to renew hand distribution, uses CFR to solve strategy
Agent3Uses RTHA to abstract the original game, uses RTHD to renew hand distribution, uses ISVE-SG and CFR to solve strategy
Agent4Uses RTHA to abstract the original game, uses RTHD to renew hand distribution, uses WSOS-SG to solve strategy
NFSPCombines deep reinforcement learning and deep learning, details provided elsewhere [24]
DeepStack1   Uses deep learning, self-play values within the continual re-solving computation, details provided elsewhere [27]
Target CFR   Uses target CFR, details provided elsewhere [18]
CFR+Uses CFR+, details provided elsewhere [42]
Compact CFR  Uses compact CFR, details provided elsewhere [17]
Deep CFR    Uses Deep CFR, details provided elsewhere [19]
## Table 5
Performance of various strategy-solving methods in two games.
PokerAgentHand
## 600090001200020000
## HUNL
Agent1 vs. HITSZ_HKL−1150.7−1260.3−1397.1−1337.4
Agent2vs. HITSZ_HKL−950.7−1060.3−997.1−937.4
## TPNL
Agent1 vs. Gibson_3p−550.7−664.3−697.1−537.4
## Agent2vs. Gibson_3p−450.3−563.1−574.8−517.2
## Table 6
Performance of the poker agent with the ISVE-SG.
PokerAgentHand
## 600090001200020000
## HUNL
Agent2 vs. HITSZ_HKL−950.7−1060.3−997.1−937.4
Agent3vs. HITSZ_HKL−350.7−254.3−257.1−217.4
## TPNL
Agent2 vs. Gibson_3p−450.3−563.1−574.8−517.2
## Agent3vs. Gibson_3p−150.7−154.3−157.197.4
## Table 7
Performance of the poker agent with the WSOS-SG.
PokerAgentHand
## 600090001200020000
## HUNL
Agent2 vs. HITSZ_HKL−950.7−1060.3−997.1−937.4
Agent3 vs. HITSZ_HKL−350.7−254.3−257.1−217.4
Agent4vs. HITSZ_HKL70.491.735.937.4
## TPNL
Agent2 vs. Gibson_3p−450.3−563.1−574.8−517.2
Agent3 vs. Gibson_3p−150.7−154.3−157.197.4
## Agent4vs. Gibson_3p250.7354.3257.1197.4
4.2.2. Component analysis of framework
Our second group of experiments tested the effect of using
ISVE-SG in games. We concluded that the estimation errors of
the intermediate state of the game affected the accuracy of the
strategy solving through the first experiment. In the traditional
algorithm, the sub-game tree was restricted within the current
rounds. Victory or defeat was judged directly based on the hand
strength of each player in the leaf node. That is, the game issued
allthepubliccardsatthecurrentleafnode,andtheneachplayer’s
hand strength was compared to decide whether they won or
lost. The game directly ended at the current node, and it did not
enter the next round. As a result, the agent tended to take the
‘‘all in’’ action or raise more chips when the hand strength was
stronger, while the action of calling or raising smaller chips from
the opponent would be interpreted as the hand strength of the
opponent being weak. Table 6 shows the performance of agent3
with the ISVE-SG.
Table 6 shows that agent3 based on the ISVE-SG was still
worsethanthecompetitionagent,buttheperformanceofagent3
was significantly better than that of the previous agents in the
first group experiments. Although these agents based on tradi-
tional methods through offline calculations were better than our
agent under the ACPC rules, the hand and the action were fixed
after abstraction, which greatly reduced the practicability. Our
method was implemented in real time and could flexibly adjust
to the actual situation.
4.2.3. Performance of agent4 based on WSOS-SG
Our third group of experiments tested the performance of
agent4 based on the WSOS-SG. Through the second experiment,
wefoundthatalthoughtheperformanceofagent3hadincreased
significantly compared with the agents in the first experiment, it
was still worse than the agent based on the traditional method.
We believe that this was largely due to the fact that the agent
receivedrelativelylittleinformationatthebeginningofthegame,
and calculating the current strategy in real time increased the
errors. To solve this problem, we further improved the agent in
the previous second experiment. The experimental results are
shown in Table 7.
Table 7 shows that the performance of agent4 based on our
WSOS-SGmethodwasbetterthantheperformancesofHITSZ_HKL
andGibson_3p,winningby37.4and197.4mbb/hin20000hands,
respectively. Our agent using the WSOS-SG method took part in
the 2018-ACPC and finished in third place for HUNL.
## 8

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
Fig. 3.Performance of agents in HUNL.
Fig. 4.Performance of agents in TPNL.
## Table 8
Performance of our poker agent agent4 in HUNL.
Poker  AgentHand
## 6000  9000   12000  20000
## HUNL
Agent4 vs. NFSP109.3  84.1   67.5   76.9
Agent4 vs. DeepStack1   2.4−57.1−15.3−21.1
Agent4 vs. Target CFR   179.7  150.1   211.1   190.5
Agent4 vs. CFR+27.2   18.4   20.1   19.3
Agent4 vs. Compact CFR  91.3   102.7   89.2   97.5
Agent4 vs. Deep CFR    10.1−17.5  5.4    7.3
Figs. 3 and 4 show the results of the agents based on our
method and the traditional method. As shown, agent4 had clear
superiority both in the HUNL and TPNL compared to the other
agents. Meanwhile, the general trend shows that the experi-
mental results were getting better and better, which indicated
that the gradual improvement of the traditional methods was
effective from the first experiment to the third experiment. Fi-
nally, the performance of the agent with our method, WSOS-SG,
was superior to that achieved by the traditional method in the
experiment.
As described above, agent4 was based on our method. To
better test its performance, we conducted a fourth group exper-
iments with six other high-level agents reported in recent years.
The experimental results are shown in Table 8.
Table 8 shows that compared with the improved CFR-based
algorithmsTargetCFR[18],CFR+[42],andCompactCFR[17],our
method had significant advantages, winning by 190.5, 19.3, 97.5
mbb/hin20000hands,respectively.Thiswasmainlybecauseour
methodintroducedareal-timeintermediatestateestimationand
blueprint strategy, which made the solution more accurate. Our
method outperformed the NFSP [24] and Deep CFR [19], winning
76.9 and 7.3 mbb/h in 20000 hands, respectively. This was be-
cause although the NFSP [24] uses a deep neural network for
strategyfitting,theoptimalstrategyrequiresdeepreinforcement
learning method to learn. Deep reinforcement learning suffers
from convergence difficulties or significant fluctuations in the
process of convergence. As a result, the learned strategy might
notreachthetheoreticaloptimum.WhiletheDeepCFR[19]uses
neural networks to estimate the regret value, and it reduces the
error caused by abstraction. Poker games are multi-round se-
quential decision games. As the game goes on, more information
will bemade public. Onlyusing a neuralnetwork to estimatethe
regret value will affect the final result.
The deep neural network is also used in the DeepStack [27],
but it uses a neural network to fit the value in each round,
which is totally different from the previous two methods that
use neural networks. This is a targeted design for poker. The
experimental results showed that our method is slightly inferior
to DeepStack [27], and our agent, agent4, lost 21.1 mbb/h in
20000 hands. However, our method also has its own advantages.
First, the deep neural network in DeepStack is used to estimate
thevalues,whichrequiresalargenumberofhigh-qualitytraining
samples to train to achieve good results, while our agent does
nothavethisrequirement.Second,ouragenthasgoodscalability
(good performance in both two- and three-player poker, while
DeepStack [27] is only suitable for two players).
We also conducted the comparison experiments with several
agents in TPNL to further verify the performance of our agent.
Therearefourcomparisonagents,Gibson_3p,ES−MCCFR,NN_3p
andOppModel. The agentES−MCCFRis based on ES-MCCFR
(external sampling MCCFR), which is also the core algorithm
of the six-player agent Pluribus. The agentNN_3pis based on
neural networks, which used a fully-connected network with
seven layers (like DeepCFR, the network takes an information set
## (observedcardsandbethistory)asinputandoutputsprobability
logits for each possible action). The agentOppModelis based on
theES-MCCFRandtheopponentmodel(heretheopponentmodel
mainly predicts the distribution probability of the opponent’s
hand).Intheexperiment,oneplayerisouragent4,theothertwo
are both the same comparison agents. Other settings are similar
tothesettingintheHUNL.Theexperimentalresultsareasshown
in Table 9.
## 9

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
## Table 9
Performance of our poker agent agent4 in TPNL.
Poker   AgentHand
## 6000   9000   12000   20000
## TPNL
Agent4 vs. Gibson_3p   250.7   354.3   257.1   197.4
Agent4 vs. ES-MCCFR   381   609.2   517.6   490.3
Agent4 vs. NN_3p501.2   590   565.1   542.8
Agent4 vs. OppModel   124   198.1   144    126.5
From Table 9, it can be found that our agent agent4 defeated
all four comparison agents in total 20000 hands respectively. It
furthershowedtheeffectivenessofouragentagent4intheTPNL.
Tobespecific,ouragentwinningtheagentGibson_3p,ES−MCCFR,
NN_3pandOppModel197.4, 490.3, 542.8 and 126.5 mbb/h in
total 20000 hands respectively. This is because for the agent
ES−MCCFR, limited by the time, it is only trained with the 2000
iterations (in each iteration, the ESMCCFR does 500 traversals).
And for the neural network of the agentNN_3p, we only refer to
the structure of DeepCFR without further detailed adjustments.
In summary, two games were used to test the effectiveness
of our method. In general, agent4 based on our method showed
excellent performance compared with other methods in com-
parative experiments. This fully demonstrated the effectiveness
of our method. In addition, in the component analysis of our
proposed method, the experimental results also verified that our
proposed RTHD and ISVE-SG were beneficial to the final solution
method.
- Conclusions and future work
We presented the method WSOS-SG based on the warm-start
online solution of sub-game to solve the imperfect-information
game. The proposed method has a strong scalability and can be
used in both two- and three-player games. In the WSOS-SG, we
propose a state value estimation method, which increases the
estimation precision of the intermediate state value of the sub-
game tree. We also proposed a real-time poker hand abstraction
(RTHA) technique, which effectively reduced the error of the ab-
straction. Our method showed superior performances compared
to some state-of-the-art methods in HUNL and TPNL poker.
Therearethreeinterestingdirectionsforfuturework.First,we
wouldliketoconsiderasolutionmethodthatrequireslessexpert
knowledge. The current method abstracts large-scale games and
then solves the abstracted game. This method requires consid-
erable domain knowledge when abstracting. One alternative is
to use a neural network to solve the game, like DeepStack [27]
and DeepCFR [19]. Second, we would like to develop a method
thatismoresuitableforcontinuousactionspaces.Atpresent,the
method for dealing with games with continuous action spaces
mainly discretize the continuous action. However, many real
gamesinvolvecontinuousactionspacescenarios.Deepreinforce-
ment learning has shown surprising performance in continuous
action space problems. It will be interesting to combine the
proposed approach with deep reinforcement learning to solve
these types of problems. Third, we would like to study more
generalizationmethods.Thesuccessofthemethodisstilllimited
to specific games, particularly poker games. However, there are
many complex problems in reality, and determining how to ex-
tend the solution method to such problems is a subject of future
work.
CRediT authorship contribution statement
Huale Li:Conceptualization, Methodology, Software, Writ-
ing - original draft.Xuan Wang:Data curation, Validation, For-
mal analysis.Kunchi Li:Software, Investigation, Methodology.
Fengwei Jia:Visualization, Validation.Yulin Wu:Software.Jiajia
Zhang:Writing - review & editing, Funding acquisition.Shuhan
Qi:Writing - review & editing, Supervision.
Declaration of competing interest
## Theauthorsdeclarethattheyhavenoknowncompetingfinan-
cial interests or personal relationships that could have appeared
to influence the work reported in this paper.
## Acknowledgment
This research was funded by key fields R&D project of Guang-
dong Province, China (No. 2020B0101380001), National Natu-
ral Science Foundation of China (No. 61902093), Natural Sci-
ence Foundation of Guangdong, China (No. 2020A1515010652),
## Shenzhen Foundational Research Funding, China Under Grant
(No. 20200805173048001), PINGAN-HITsz Intelligence Finance
ResearchCenter,China,Ricoh-HITszJointResearchCenter,China,
GBase-HITsz Joint Research Center, China.
## References
[1] D. Tolpin, S.E. Shimony, MCTS based on simple regret, in: Twenty-Sixth
AAAI Conference on Artificial Intelligence, 2012.
[2] C.B. Browne, E. Powley, D. Whitehouse, S.M. Lucas, P.I. Cowling, P. Rohlf-
shagen, S. Tavener, D. Perez, S. Samothrakis, S. Colton, A survey of Monte
Carlo tree search methods, IEEE Trans. Comput. Intell. AI Games 4 (1)
## (2012) 1–43.
## [3] V. Mnih, K. Kavukcuoglu, D. Silver, A. Graves, I. Antonoglou, D. Wierstra,
M. Riedmiller, Playing atari with deep reinforcement learning, Comput. Sci.
## (2013).
[4] V. Mnih, K. Kavukcuoglu, D. Silver, A.A. Rusu, J. Veness, M.G. Bellemare,
A. Graves, M. Riedmiller, A.K. Fidjeland, G. Ostrovski, et al., Human-level
control through deep reinforcement learning, Nature 518 (7540) (2015)
## 529.
[5] D. Silver, A. Huang, C.J. Maddison, A. Guez, L. Sifre, G. Van Den Driessche,
J. Schrittwieser, I. Antonoglou, V. Panneershelvam, M. Lanctot, et al.,
Mastering the game of go with deep neural networks and tree search,
## Nature 529 (7587) (2016) 484.
[6] D. Silver, J. Schrittwieser, K. Simonyan, I. Antonoglou, A. Huang, A. Guez, T.
Hubert, L. Baker, M. Lai, A. Bolton, et al., Mastering the game of go without
human knowledge, Nature 550 (7676) (2017) 354.
## [7] D. Silver, T. Hubert, J. Schrittwieser, I. Antonoglou, M. Lai, A. Guez, M.
## Lanctot, L. Sifre, D. Kumaran, T. Graepel, T. Lillicrap, K. Simonyan, D.
Hassabis, A general reinforcement learning algorithm that masters chess,
shogi, and go through self-play, Science 362 (6419) (2018) 1140–1144.
[8] J. Nash, Non-cooperative games, Ann. of Math. (1951) 286–295.
[9] M. Zinkevich, M. Johanson, M. Bowling, C. Piccione, Regret minimization in
games with incomplete information, in: Advances in Neural Information
Processing Systems, 2008, pp. 1729–1736.
[10] A. Gilpin, T. Sandholm, Finding equilibria in large sequential games of
imperfect information, in: Proceedings 7th ACM Conference on Electronic
Commerce, 2006, pp. 160–169.
## [11] M. Johanson, N. Bard, M. Lanctot, R. Gibson, M. Bowling, Efficient Nash
equilibrium approximation through Monte Carlo counterfactual regret
minimization, in: International Conference on Autonomous Agents &
## Multiagent Systems, 2012.
[12] V. Lisy, M. Lanctot, M. Bowling, Online Monte Carlo counterfactual regret
minimization for search in imperfect information games, in: International
Conference on Autonomous Agents & Multiagent Systems, 2015.
[13] N. Burch, M. Bowling, Cfr-d: Solving imperfect information games using
decomposition, 2013, pp. 1–15.
[14] J. Zhang, H. Liu, Reinforcement learning with monte carlo sampling in
imperfect information problems, in: International Conference on Cognitive
Computing, Springer, 2018, pp. 55–67.
[15] O. Tammelin, Solving large imperfect information games using cfr+, 2014,
ArXiv Preprint:1407.5042.
[16] T.S. Noam Brown, Reduced space and faster convergence in imperfect-
information games via pruning, in: Proceedings of the 34th International
Conference on Machine Learning, 2017, pp. 596–604.
[17] E.G. Jackson, Compact CFR, in: Workshops At the Thirtieth AAAI Conference
on Artificial Intelligence, 2016.
[18] E.G. Jackson, Targeted CFR, in: Workshops At the Thirty-First AAAI
Conference on Artificial Intelligence, 2017.
## 10

H. Li, X. Wang, K. Li et al.Knowledge-Based Systems 231 (2021) 107434
[19] N. Brown, A. Lerer, S. Gross, T. Sandholm, Deep counterfactual regret
minimization, in: International Conference on Machine Learning, 2019.
[20] D. Billings, A. Davidson, J. Schaeffer, D. Szafron, The challenge of poker,
## Artificial Intelligence 134 (1) (2002) 201–240.
[21] D. Billings, N. Burch, A. Davidson, R.C. Holte, J. Schaeffer, T. Schauenberg,
D. Szafron, Approximating game-theoretic optimal strategies for full-scale
poker, in: International Joint Conference on Artificial Intelligence, 2003.
[22] K. Hu, S. Ganzfried, Midgame solving: A new weapon for efficient large-
scale equilibrium approximation, in: 29th International Conference on
Tools with Artificial Intelligence, 2017.
[23] R. Gibson, D. Szafron, Regret minimization in multiplayer extensive games,
in: Twenty-Second International Joint Conference on Artificial Intelligence,
## 2011.
[24] J. Zhang, H. Liu, Reinforcement learning with Monte Carlo sampling in
imperfect information problems, in: International Conference on Cognitive
## Computing, 2018.
[25] J. Shi, M.L. Littman, Abstraction methods for game theoretic poker, in:
International Conference on Computers and Games, Springer, 2000, pp.
## 333–345.
[26] A. Gilpin, T. Sandholm, A competitive Texas Hold’em poker player via auto-
mated abstraction and real-time equilibrium computation, in: International
Joint Conference on Autonomous Agents & Multiagent Systems, 2006.
## [27] M. Moravčík, M. Schmid, N. Burch, V. Lis
## `
y, D. Morrill, N. Bard, T.
Davis, K. Waugh, M. Johanson, M. Bowling, Deepstack: Expert-level ar-
tificial intelligence in heads-up no-limit poker, Science 356 (6337) (2017)
## 508–513.
[28] N. Brown, T. Sandholm, Safe and nested subgame solving for imperfect-
informationgames,in:AdvancesinNeuralInformationProcessingSystems,
2017, pp. 689–699.
[29] N. Brown, T. Sandholm, Superhuman ai for heads-up no-limit poker:
Libratus beats top professionals, Science 359 (6374) (2017) 1733.
[30] N. Brown, T. Sandholm, Superhuman ai for multiplayer poker, Science 365
## (6456) (2019) 885–890.
[31] R.G. Gibson, Regret minimization in games and the development of
champion multiplayer computer poker-playing agents, 2014.
[32] R. Cressman, C. Ansell, K. Binmore, Evolutionary Dynamics and Extensive
Form Games, vol. 5, MIT Press, 2003.
[33] L.K. Simon, M.B. Stinchcombe, Extensive form games in continuous time:
Pure strategies, Econometrica (1989) 1171–1214.
[34] N. Brown, T. Sandholm, Solving imperfect-information games via dis-
counted regret minimization, in: Proceedings of the AAAI Conference on
Artificial Intelligence, Vol. 33, pp. 1829–1836.
[35] J. Zhang, L. Hong, Building endgame data set to improve opponent mod-
eling approach, in: IEEE Second International Conference on Data Science
in Cyberspace, 2017.
[36] R. Gibson, Regret minimization in non-zero-sum games with applications
to building champion multiplayer computer poker agents, 2013, arXiv
preprint arXiv:1305.0034.
[37] K. Waugh, A fast and optimal hand isomorphism algorithm, in: Workshops
At the Twenty-Seventh AAAI Conference on Artificial Intelligence, 2013.
[38] A. Likas, N. Vlassis, J.J. Verbeek, The global k-means clustering algorithm,
## Pattern Recognit. 36 (2) (2003) 451–461.
[39] T. Denoeux, Analysis of evidence-theoretic decision rules for pattern
classification, Pattern Recognit. 30 (7) (1997) 1095–1107.
[40] A. Krizhevsky, I. Sutskever, G.E. Hinton, ImageNet classification with deep
convolutional neural networks, in: International Conference on Neural
## Information Processing Systems, 2012.
[41] L. Kocsis, C. Szepesvári, Bandit based monte-carlo planning, in: European
Conference on Machine Learning, Springer, 2006, pp. 282–293.
[42] N. Burch, M. Moravcik, M. Schmid, Revisiting CFR+ and alternating updates,
## Artificial Intelligence 64 (2019) 429–443.
## 11