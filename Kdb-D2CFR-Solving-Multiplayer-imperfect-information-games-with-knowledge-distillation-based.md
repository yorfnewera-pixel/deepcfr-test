

Knowledge-Based Systems 272 (2023) 110567
Contents lists available at ScienceDirect
Knowledge-BasedSystems
journal homepage: www.elsevier.com/locate/knosys
Kdb-D2CFR:SolvingMultiplayerimperfect-informationgameswith
knowledgedistillation-basedDeepCFR
HualeLi
a,b,c,d
,ZengyueGuo
c
,YangLiu
c
,XuanWang
c
,ShuhanQi
c,d,e,
## ∗
,JiajiaZhang
c,
## ∗
## ,
JingXiao
f
a
School of Software, Northwestern Polytechnical University, 710072, Xi’an, China
b
Yangtze River Delta Research Institute of NPU, Taicang, 215400, China
c
School of Computer Science and Technology, Harbin Institute of Technology Shenzhen, 518055, China
d
Guangdong Provincial Key Laboratory of Novel Security Intelligence Technologies, 518000, Shenzhen, China
e
## Peng Cheng Laboratory, 518000, Shenzhen, China
f
Ping An Insurance (Group) Company, 518000, Shenzhen, China
a r t i c l e    i n f o
Article history:
## Received 9 August 2022
Received in revised form 4 April 2023
## Accepted 10 April 2023
Available online 18 April 2023
## Keywords:
## Game
## Imperfectinformation
## Counterfactualregretminimization
## Deeplearning
## Knowledgedistillation
a b s t r a c t
Counterfactual regret minimization (CFR) is a popular method for finding approximate Nash equilib-
rium in imperfect-information games (IIG). However, CFR based methods for the IIG are either only
designed for two-player IIGs or require much expert knowledge. In this paper, towards to solve the
multiplayerIIGproblemwithoututilizingmuchexpertknowledge,weproposedapracticalknowledge
distillation based framework, which aims to transferring the knowledge from model of two-player
IIG into the multiplayer one. By this framework, both of the training efficiency and performance is
improved. To eliminate the requirement of expert knowledge, here we introduced a deep learning
based CFR in the framework, by which the counterfactual value of CFR can be estimated in an end-
to-end way without any expert knowledge and abstraction. We further propose kdb-DeepCFR and
kdb-D2CFR based on DeepCFR and D2CFR respectively, which can effectively solve the strategy of
multiplayer large-scale game problems. The extensive experiments conducted on 3-8 players poker
games suggest that our method outperforms other baselines in the game performance.
©2023ElsevierB.V.Allrightsreserved.
## 1. Introduction
As a research domain of artificial intelligence, computer game
mainly studies the theory of intelligent decision-making [1–3].
## Gameshavebeenregardedasthetouchstonetotestthedevelop-
ment of artificial intelligence. Generally, compared with perfect-
informationgame(PIG),imperfect-informationgame(IIG)ismore
complex because it contains private information unobservable
to other players. For example, poker game has already served
as a popular benchmark of the IIG [4–12], since it contains all
necessary elements of the IIG.
For the two-player IIG, a typical solution is to find its Nash
equilibrium or approximate Nash equilibrium. A Nash equilib-
rium is a list of strategies where no player can improve by
deviating to a different strategy [13]. The counterfactual regret
minimization(CFR)[14]anditsvariantmethods[11,15,16],which
aim to find such strategies with approximate Nash equilibrium,
have already achieved great success in the field of IIG. However,
## ∗
Corresponding authors.
E-mail addresses:shuhanqi@cs.hitsz.edu.cn (S. Qi), zhangjiajia@hit.edu.cn
(J. Zhang).
CFR has to traverse the whole game tree, which leads to high
computational and storage complexity. Such problem limits its
further application in large-scale IIGs. The traditional solution
is to abstract the original large-scale game into a small-scale
game, then solving the strategy of the small-scale game with
CFR. In this way, although the large-scale IIG can be solved
with CFR, this solution requires much expert knowledge and
causescertaininformationloss.ArecentmethodDeepCFRgreatly
reduces the requirement of expert knowledge by applying deep
neural networks to CFR [17]. By training the DeepCFR in an end-
to-end way, IIGs with two-player can be solved directly without
anyabstraction.DoubleNeuralCFR(DNCFR)[18],singleDeepCFR
(SD-CFR) [19] and deep dueling CFR (D2CFR) [20] are all deep
learning[21–23]basedmethodssimilartotheDeepCFRinrecent
years.
Although there have been many successful studies in IIGs,
it should be noted that they only focus on two-player games
[24–28]. For the IIG with multiplayers (eg. Poker games with
eight players), which is usually much more complex than two-
playerIIGs,howtosolveiteffectivelyisstillaproblemworthyof
study.Althoughdeepreinforcementlearningbasedmethodshave
achieved successes in some games like StarCraft, Dota and Honor
https://doi.org/10.1016/j.knosys.2023.110567
0950-7051/©2023 Elsevier B.V. All rights reserved.

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Fig. 1.The neural network architecture of DeepCFR. The network takes an infoset (observed cards and bet history) as input and outputs values (advantages or
probability logits) for each possible action. The cards are represented as the sum of three embeddings: a rank embedding (1–13), a suit embedding (1–4), and a card
embedding (1–52). Each betting position is encoded by a binary value specifying whether a bet has occurred, and a float value specifying the bet size.
ofKings[29–31].However,itsnetworktrainingneedslarge-scale
computingsupport,anditsmodelisnotinterpretable.Inaddition,
reinforcement learning, especially deep reinforcement learning,
requires complex super parameter tuning during training, such
as learning rate, discount factor, cache size, etc; Moreover, its
convergenceisnotonlylackoftheoreticalguarantee,butalsothe
trainingofthenetworkneedsextracomputingpower.Forexam-
ple,AlphaStar[31]needs16TPUsfordatasamplingandnetwork
training. In the research on poker games, CFR based methods
are the most popular and dominant, with unprecedented suc-
cess, such as DeepStack [15], Libratus [16] and Pluribus [32].
Pluribus [32] is first superhuman agent in six-player no-limit
Texas hold’em game, which solves the game through the combi-
nation of abstraction technique and CFR methods. It only focuses
on solving six-player games and it needs to be redesigned when
thenumberofplayerchanges.DuetothefactthatfindingaNash
equilibrium in multiplayer IIGs is at least very hard and even
approximating a Nash equilibrium is hard [33]. Thus the goal of
solvingmultiplayerIIGschangesfromfindingaNashequilibrium
to improving its game performance as much as possible. Despite
the game performance of Pluribus surpasses that of professional
players, it requires much expert knowledge for detailed design
andabstractiontechniques,whichisagreatdeficiencyforfurther
research.
Inthispaper,westudytheproblemofIIGwithvariantnumber
of players without any expert knowledge and abstraction tech-
nique. Following with Pluribus, here we also aim to improve the
game performance of the solving method, instead of approxi-
mating a Nash equilibrium with theoretical guarantee. To solve
the cold-start training problem of vary number of players in
IIG, a method of reusing the knowledge of two-player game to
multiplayers is proposed. Here the motivation is that in many
cases, although the number of players changes, much knowledge
in two-player games can be utilized to warm up the training for
multiplayer.Thusweproposeaknowledgedistillationbasedmul-
tiplayerIIGmethodthatcanlearnthestrategyfromatwo-player
IIG model. We summarize our contributions as follows:
- We design a framework studying on solving the strategy
of multiplayer IIGs, which combines CFR based methods
and knowledge distillation. The framework can make the
knowledge in two-player IIGs shift to multiplayer IIGs by
knowledge distillation.
- We present kdb-DeepCFR and kdb-D2CFR, which can re-
duce the requirement of expert knowledge in traditional
CFR methods. To the best of our knowledge, the kdb-
DeepCFR and kdb-D2CFR are the first DeepCFR based
method used to solve multiplayer IIGs.
- Extensive experimental results show that the proposed
methodperformsbetterthanotherbaselinesinbothofthe
two-player and multiplayer IIGs.
The rest of our paper is organized as follows. In Section 2,
concepts of extensive-form game, DeepCFR and knowledge dis-
tillation are introduced. Our proposed method is described in
Section 3, which includes details of the kdb-DeepCFR and kdb-
D2CFR. In Section 4, we evaluate the performance of the kdb-
DeepCFR and kdb-D2CFR. Finally, in Section 5, we present the
conclusions of the paper.
## 2. Background
2.1. Extensive-form game
An extensive-form game is a model of game that describes
sequential-decisions in the field of IIGs. Formally, a finite
extensive-form game contains six components
## ⟨
N,H,P,f
c
,I,u
i
## ⟩
## [34]:
## ⟨
N,H,P,f
c
,I,u
i
## ⟩
: playerirepresents a finite setNof game
players,N={1,2,...,n}.Anode(i.e.,history)hisdefinedbyall
informationofthecurrentsituation,includingprivateknowledge
knowntoonlyoneplayer.AfinitesetHofsequences,thepossible
histories of actions, such that the empty sequence is inHand
everyprefixofasequenceinHisalsoinH.A(h)={a|(h,a)∈H}
are actions available after a nonterminal historyh∈H.Z⊆H
are terminal histories.Pis the player function.P(h) is the player
takingactionaafterhistoryh.P(h)=crepresentsthatthechance
determinestheactionafterhistoryh.Afunctionf
c
thatassociates
with every historyhfor whichP(h)=ca probability measure
f
c
(·|h) onA(h). The set I
i
## ∈I
i
is an information set of playeri. For
any information setI
i
, all nodesh,h
## ′
## ∈I
i
are indistinguishable to
playeri. Payoff functionu
i
defines the payoff of terminal statez
for each playeri. For a zero-sum game, there isu
## 1
## +u
## 2
## =0.
In an extensive-form game, a strategyσ
i
(I) of playeriis a
probability vector over actions on information setI. A set of
strategies for players,σ
## 1
## ,σ
## 2
## ,...,σ
n
, makes up a strategy profile
σ, andσ
## −i
represents the strategy inσexcept the strategyσ
i
of
playeri. In addition,π
σ
(h) is the probability withhoccurs if all
players play according to the strategyσ, andπ
σ
## (I)=Σ
h∈I
π
σ
## (h).
π
σ
i
(h) is the contribution of playerito this probability. And
formally,π
σ
i
## (h)=
## ∏
i∈N∪{c}
π
σ
i
(h).Accordingly,π
σ
## −i
## (h)ofhistoryh
isthecontributionofallplayers(includingchanceplayer)except
playeri.
2.2. DeepCFR
DeepCFR [17] is an improved variant of CFR, which obviates
the need for abstraction by instead using deep neural networks
to approximate the behavior of CFR in the full game. The neural
network architecture of DeepCFR is shown in Fig. 1. DeepCFR
computes an approximation of Nash equilibrium by applying the
following two steps:
## 2

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Step 1: Calculate the advantage value. Compared with tradi-
tional CFR-based methods, the value network is used to fit the
advantagevalueofplayerioneachiteration.Theadvantagevalue
## D
## T
i
(I,a) in DeepCFR is defined as:
## D
## T
i
(I,a)=
## R
## T
i
(I,a)
## ∑
## T
t=1
## (tπ
σ
t
## −i
## (I))
## (1)
wheretis the number of iteration,Tis the total number of
iteration.iis the player and−iis the opponents except the
playeri.σ
t
is the strategy on iterationt.R
## T
i
(I,a) is cumulative
regret of actionaon information setI. In DeepCFR,R
## T
i
(I,a)=
## ∑
## T
t=1
## (tr
t
i
(I,a)),r
t
i
(I,a)istheinstantregretofplayeriforactiona
oninformationsetI.Forinstantregret,r
t
i
(I,a)=v
σ
t
(I,a)−v
σ
t
## (I),
and counterfactual valuev
σ
t
(I) of playeriat information setI
is the expected payoff to playeriwhen reachingI, weighted by
the probability that playeriwould reached information setIif
she tried to do so that iterationt. Counterfactual action value
v
σ
t
(I,a) is the same except it assumes that playeriplays action
aat information setIwith 100% probability.
Step2:Updatethestrategyofnextiteration.CFR-basedmeth-
ods are iterative method and DeepCFR is no exception. In Deep-
CFR, regret matching algorithm [35] is used to calculate the
strategy of next iterationt+1 and can be described as:
σ
t+1
i
(I,a)=
## ⎧
## ⎨
## ⎩
## D
t,+
i
(I,a)
## ∑
a
## ′
## ∈A(I)
## D
t,+
i
## (
## I,a
## ′
## )
## ,
## ∑
a∈A(I)
## D
t,+
i
(I,a)>0
## 1
## |A(I)|
## ,otherwise
## (2)
whereD
t,+
i
(I,a)=max
## (
## D
t
i
(I,a),0
## )
## .
In DeepCFR, the training data ofD(I,a) comes from the exter-
nal sampling MCCFR (ES-MCCFR) [36], which traverses the game
tree and produces the instant regret values. All training data are
storedinamemorybufferB
v
i
## ,whichisupdatedthroughreservoir
sampling [37] when its maximum capacity is reached. In addi-
tion, DeepCFR fits a policy network to approximate the average
strategy ̄σ
## T
i
(I,a)=
## ∑
## T
t=1
## (tπ
σ
i
(I)σ
t
(I,a))
## ∑
## T
t=1
## (tπ
σ
i
## (I))
, when the total iterationTis
reached.
2.3. Knowledge distillation
The concept of knowledge distillation is put forward by Hin-
ton in 2015 [38], which is to realize the knowledge transfer
from complex model to simplified model. The traditional knowl-
edge distillation model ‘‘teacher-student’’ is mainly composed of
teacher network and student network, as shown in Fig. 2. The
modelof‘‘teacher-student’’ismainlydividedintotwosteps:first,
trainateachermodelwithstrongperformance.Thismodelisgen-
erally complex, which can be a large and deep network model or
amodelaftertheintegrationofmultiplemodels;Then,underthe
guidanceoftheteachermodel,trainthesimplifiedstudentmodel,
and introduce the soft target related to the teacher model as a
partoftheoveralllossfunction,soastoinducethestudentmodel
totrainandlearn,andfinallyrealizetheknowledgetransferfrom
the teacher model to the student model.
- Our method
In this section, we first describe the overview of the pro-
posed framework. Then, the detail of proposed kdb-DeepCFR and
kdb-D2CFR will be introduced.
Fig. 2.The traditional architecture of knowledge distilling ‘‘Teacher-Student’’.
Fig. 3.The framework of Kdb-DeepCFR. The Kdb-DeepCFR is composed of two
models: Teacher model and Student model. The two models are both based on
the model of DeepCFR.
3.1. An overview of the proposed framework
The above mentioned DeepCFR in Section 2 is proposed only
for two-player IIGs. Here we discuss how to adopt the DeepCFR
into the IIG with multiplayers. Considering that when dealing
withthesametypeofgames,suchaspokergames,fortwo-player
games and multiplayer games, even if the number of players has
increased, there are some common knowledge in the IIGs with
different scale. For example, one player with the hand ‘AA’ has
a high probability of choosing to raise, whether in two-player
or multiplayer poker games. In other words, we can utilize the
knowledge of two-players IIG to help training the strategy of
multiplayer IIGs. Specifically, in our method, we use knowledge
distillation to transfer the knowledge of two-player game to
multiplayer game so as to warm up the training of multiplayer
game.
In this paper, we propose a framework, knowledge distilling-
based DeepCFR (Kdb-DeepCFR), which can transfer the knowl-
edge of DeepCFR in two-player IIGs to solving multiplayer IIGs.
TheframeworkofKdb-DeepCFRisshowninFig.3,whichiscom-
posedoftwomodels:TeachermodelandStudentmodel.Thetwo
modelsarebothbasedonthemodelofDeepCFR.Slightlydifferent
from the DeepCFR, we divided its network in the first few layers,
which is mainly to facilitate the Block1 of the Teacher model
to guide the Block2 of the Student model. It should be noted
that, different from the traditional ‘‘Teacher-student’’ knowledge
## 3

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Fig. 4.The framework of Kdb-D2CFR. The Kdb-D2CFR is also composed of two
models: Teacher model and Student model. The two models are both based on
the model of D2CFR.
distillationmethods,theknowledgeintheteachermodelistrans-
ferred from two parts to the student model: from the feature
block Block1 to Block2 and from the output of the teacher model
to the student model respectively.
Furthermore,similartotheproposedkdb-DeepCFR,wefurther
propose knowledge distilling-based D2CFR (Kdb-D2CFR), which
can transfer the knowledge of D2CFR in two-player IIGs to solv-
ing multiplayer IIGs. The framework of Kdb-D2CFR is shown in
Fig.4,whichisalsocomposedoftwomodels:Teachermodeland
Student model. The two models are both based on the model of
D2CFR, which is the same with kdb-DeepCFR.
Two methods are proposed in this paper, both of which are
dedicated to solving multi-player strategy in the field of IIGs.
Firstly, they are both strategy solving methods using neural net-
works combined with CFR, which are a kind of end-to-end strat-
egy solving method from state input to strategy output. Sec-
ondly, their solution frameworks are designed based on tradi-
tional knowledge distillation, which add feature layer distillation
to the classical teacher-student distillation model. Thirdly, the
loss functions of their student models are also similar. Finally,
theycaneffectivelysolvestrategieswith3–8gameplayersinthe
field of IIGs. Of course, while they have the above similarities,
they also have some differences. Firstly, the training methods of
value networks in their respective teacher models are different.
Secondly, the network structure of the value network in their
teacher models is also different. Kdb-DeepCFR uses a fully con-
nected network, while Kdb-D2CFR uses a decoupled network.
Finally,theirteachermodelshavedifferentfocus.Kdb-DeepCFRis
directly fitting regret values through a fully connected network,
while Kdb-D2CFR is fitting regret values by focusing on states
with higher value through a decoupled network.
3.2. The detail of kdb-DeepCFR and kdb-D2CFR
We will mainly introduce the sample collection and network
training in the training process of kdb-DeepCFR. First of all, for
the ‘‘teacher’’ model in kdb-DeepCFR, the samples required for
training are collected by using DeepCFR in the iterative solution
of the two-player imperfect-information games.
Specifically,firstly,ineachiterationprocess,ES-MCCFRisused
to traverse the game tree of the specific game. The informa-
tion set, action regret value samples generated in the traversal
process are stored in the value network sample pool. Secondly,
updatetheiterationstrategyofthenextiterationthroughtheRM.
## Thirdly,expandthegametreeaccordingtotheupdatingstrategy
in the next iteration. Store the information set, action samples
generated in the iteration process into the strategy sample pool.
Considering that DeepCFR needs to approach the value network
and strategy network respectively during training, the sample
collection here also collects the value samples and strategy sam-
plesrespectively.Whenthestoredsamplesreachtheupperlimit
of the sample pool capacity, the reservoir sampling technique is
used to update the sample pool.
For the ‘‘student’’ model in kdb-DeepCFR, its samples are col-
lected during the strategy solving process of multiplayer games.
## Similartothesamplecollectionprocessrequiredbythe‘‘teacher’’
model, the only difference is that the ‘‘teacher’’ model collects
samples in the process of solving the two-player game strategy,
while the ‘‘student’’ model collects samples in the process of
solving the multiplayer game strategy.
Training the kdb-DeepCFR:Since the two proposed methods
kdb-DeepCFRandkdb-D2CFRarebasicallysimilar,wewillmainly
introducethetrainingofkdb-DeepCFRhere.Thenetworktraining
of Kdb-DeepCFR is carried out from the teacher model and the
studentmodel.Fortheteachermodel,itsnetworktrainingcanbe
divided into the training of value network and strategy network
respectively. For the value network of the teacher model, its loss
functionL
## (
θ
i
## )
istheMSE(meansquarederror)[39]betweenthe
output of value network and the samples through traversing the
game tree with ES-MCCFR, which can be depicted as:
## L
## (
θ
i
## )
## =E
## (
## I,t
## ′
## ,
## ̃
r
## ′
## )
## ∼M
## V,i
## [
t
## ′
## ∑
a
## (
## ̃
r
t
## ′
(I,a)−V
## (
## I,a|θ
i
## )
## )
## 2
## ]
## (3)
wheret
## ′
is the number of iteration,
## ̃
r
t
## ′
(I,a) is the regret value
of actionaon the information setIatt
## ′
iteration,V
## (
## I,a|θ
i
## )
is
the approximate value of value network for the regret value of
actionaontheinformationsetI,M
## V,i
isthesampledataofvalue
network.
For the policy network of the teacher model, its loss function
## L
## (
θ
## Π
## )
is the MSE between the output of the policy network and
the actual iteration policy, which can be depicted as:
## L
## (
θ
## Π
## )
## =E
## (
## I,t
## ′
## ,σ
## ′
## )
## ∼M
## Π
## [
t
## ′
## ∑
a
## (
σ
t
## ′
(I,a)−Π
## (
## I,a|θ
## Π
## )
## )
## 2
## ]
## (4)
whereΠ
## (
## I,a|θ
## Π
## )
is the probability of actionaon the infor-
mation setIfor the policy network,σ
t
## ′
(I,a) is the probability of
actionaon the information setIfrom the iteration strategyσ
t
## ′
## ,
σ
t
## ′
is the sample data of policy network.
For the student model, the training process is similar with
teacher model except that the guidance of the teacher model is
added to the loss function. Specifically, for the value network of
the student model, its loss functionL
total
## (
θ
i
## )
can be depicted as:
## L
total
## (
θ
i
## )
## =L
## (
θ
i
## )
+λL
block
## (
θ
i
## )
+γL
output
## (
θ
i
## )
## (5)
whereL
total
## (
θ
i
## )
is the total loss function of the value network,
## L
## (
θ
i
## )
is the hard loss function between the student model and
thesamplelabel.L
block
## (
θ
i
## )
isthelossfunctionbetweentheoutput
of Block1 and Block2.L
output
## (
θ
i
## )
is the loss function between the
output of the teacher model and the student model.λandγare
hyperparameters.
The loss functionL
block
## (
θ
i
## )
between the output of Block1 and
Block2 can be depicted as:
## L
block
## (
θ
i
## )
## =E
## (
## I,t
## ′
## ,
## ̃
r
## ′
## )
## ∼M
## V,i
## [
## (
## W
block1
## (x;θ
## Te
## )−W
block2
## (x;θ
## Stu
## )
## )
## 2
## ]
## (6)
whereW
block1
## (x;θ
## Te
) is the output of the Block1 with the pa-
rameterθ
## Te
## ,W
block2
## (x;θ
## Stu
) is the output of the Block2 with the
parameterθ
## Stu
## .
## 4

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
## Thelossfunctionbetweentheoutputoftheteachermodeland
the student modelL
output
## (
θ
i
## )
can be depicted as:
## L
output
## (
θ
i
## )
## =H(P
## Te
## ,P
## Stu
## )(7)
whereHis cross entropy,P
## T
## Te
## =softmax(
a
## Te
## T
) andP
## T
## Stu
## =
softmax(
a
## Stu
## T
) are the probabilities output by softmax from the
value network of the teacher model and student model.a
## Te
and
a
## Stu
are the output of the teacher model and student model,Tis
the temperature coefficient.
For the policy network of the student model, similar with the
lossfunctionofthevaluenetwork,thelossfunctionL
total
## (
θ
## Π
## )
can
be depicted as:
## L
total
## (
θ
## Π
## )
## =L
## (
θ
## Π
## )
+λL
block
## (
θ
## Π
## )
+γL
output
## (
θ
## Π
## )
## (8)
ThelossfunctionL
block
## (
θ
## Π
## )
ofthepolicynetworkbetweenthe
output of Block1 and Block2 can be depicted as:
## L
block
## (
θ
## Π
## )
## =E
## (
## I,t
## ′
## ,
## ̃
r
## ′
## )
## ∼M
## V,i
## [
## (
## W
block1
## (x;θ
## Te
## )−W
block2
## (x;θ
## Stu
## )
## )
## 2
## ]
## (9)
The loss function of the polict network between the output
of the teacher model and the student modelL
output
## (
θ
## Π
## )
can be
depicted as:
## L
output
## (
θ
## Π
## )
## =H(P
## Te
## ,P
## Stu
## )(10)
where all parameters are similar to those in the value network
settings above.
Training the kdb-D2CFR:Theaboveismainlyanintroduction
to the kdb-DeepCFR for solving the game strategy of multiplayer
imperfect-information games based on knowledge distillation.
Consideringthatkdb-DeepCFRhasbeenintroducedindetailfrom
two aspects: sample collection and network training. And the
general framework of kdb-D2CFR and kdb-DeepCFR is basically
similar, here we only give an overview of this method.
In terms of sample collection, kdb-D2CFR is exactly the same
as that in the kdb-DeepCFR. It also collects value network sam-
ples by partially traversing the game tree through ES-MCCFR at
each iteration. Then update the iterative strategy through the
RM, collect the strategy network samples and store them in the
sample pool. In terms of network training, although there are
differences in network structure between D2CFR and DeepCFR,
kdb-D2CFR still separates the value network in the early feature
layeraccordingtotheframeworkofkdb-DeepCFR.Anddistillsthe
knowledge of the early feature layer and the later output layer
through knowledge distillation, so as to achieve the knowledge
transfer from the ‘‘teacher’’ model to the ‘‘student’’ model. In
addition, the network training and model testing of kdb-D2CFR
are no different from that of kdb-DeepCFR, thus we do not carry
out it in detail in the paper.
## 4. Experiments
In this section, the experimental setup and experimental re-
sultsareintroduced.Thetestbed,implementationandevaluation
metric will be detailed described in the experimental setup. The
comparison experiments and ablation studies are conducted in
the experimental results.
4.1. Experimental setup
Experimental testbed:Poker is a family of games that in-
cludes hidden information, which has been regarded as the most
classic game benchmarks in IIGs in recent years. Many success-
ful CFR-based methods and applications take poker games as
the testbed to verify their effectiveness [40–44], such as Deep-
## Stack [15], Libratus [16], Pluribus [32].
In this paper, the Heads-up No-Limit Texas hold’em (HUNH)
and multiplayer No-Limit Texas hold’em (M-NH) are used to test
theeffectivenessoftheproposedmethod.TheHUNHcontains52
cards in total and consists of four rounds of betting. Each player
has two private hands in the first round and totally five public
cards in next three rounds. The four betting rounds are preflop,
flop,turnandriver.Threekindsofactions,fold,callandraisecan
bechosenbyeachplayeronaroundofbetting.M-NHareexactly
the same with HUNH except for the number of game players.
Implementation detail:All experiments in this paper are
conducted on the platform OpenSpiel [45], which is a collection
of environments and algorithms for research in computer games.
ThecomparisonalgorithmDeepCFRistrainedcompletelyaccord-
ing to the algorithm provided by the OpenSpiel. All parameters
are set completely according to the original paper.
We set hyperparameters as follow. For the value network, the
information sets are taken as input and outputs the regret value
of each action. The policy network outputs probability of each
legal actions. The batch size is 200. The parameters are updated
by Adam optimizer [46] with a learning rate 0.001. The memory
capacityis100,000.FortheKdb-DeepCFR,thenumberofiteration
is500,whichisenoughtoverifytheeffectivenessofthemethod.
In addition, all experiments are conducted on four Xeon(R) CPUs
of E5-2640 with 10 cores @2.40 GHz, and one Tesla P100 GPU
with 16G memory.
Evaluation metric:In this paper, the effectiveness of our
method will be evaluated with a popular metric in the field
of IIGs: game performance, just like the experimental setup in
previous works [15,16,45].
4.2. Experimental results
In this section, comparison experiments and ablation studies
are conducted. For comparison experiments, Head-up no-limit
Texas hold’em poker and multiplayer no-limit Texas hold’em
poker are used to test the effectiveness Kdb-DeepCFR and Kdb-
D2CFR respectively. For ablation studies, the component of Kdb-
DeepCFR and Kdb-D2CFR are ablated respectively.
4.2.1. Comparison experiments
The experiment is to verify the effectiveness of Kdb-DeepCFR
andKdb-D2CFRontheM-NH.Specifically,thisgroupexperiment
is conducted on the number of player from 3 to 8 respectively.
For convenience, three-player no-limit Texas hold’em poker is
represented by 3-NH in the following, other multiplayer no-limit
Texas hold’em poker games are represented in a similar form.
There are few multiplayer baselines available at present in
poker games, and since many technical details are lost in their
paper (eg. Pluribus [32]), even re-implement some baselines is
vary difficult. In this paper four comparison methods the Deep-
CFR [17], D2CFR [20], DREAM [47] and the agent Jaysen [48] are
used in the comparison experiment. It should be noted that the
agent here we extend DeepCFR to multiplayer game and Jaysen
is the3rd prizein 2018-ACPC (Annual Computer Poker Compe-
tition, www.computerpokercompetition.org), which is only used
on the 6-NH [48] and based on the EHS (effective hand strength)
technique [4]. The number of test game is 10,000. The game
position will be randomly initialized each time. The game mode
adopts one proposed algorithm and multiple same comparison
algorithms to match. TheY-axis is the cumulative winning of
the proposed algorithm. The positive cumulative winning of the
proposed algorithm shows that its game performance is better
than the comparison algorithm. The experimental results are
shown in Figs. 5–7.
The experimental results on the M-NH of Kdb-DeepCFR vs.
DeepCFR and Kdb-DeepCFR vs. DREAM are shown in Figs. 5(a)
## 5

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Fig. 5.Game experiment results of Kdb-DeepCFR. TheX-axis represents the number of played game, theY-axis represents the corresponding cumulative winning.
The higher the curve, the better the method. The six curves in the figure represents the results tested on various number of game players. For example, ‘‘3-players’’
in left subfigure represents the results of Kdb-DeepCFR vs. DeepCFR on the three-player test game.
Fig. 6.Game experiment results of Kdb-D2CFR. TheX-axis represents the number of played game, theY-axis represents the corresponding cumulative winning. The
higher the curve, the better the method.
and 5(b) respectively. From the curve of cumulative winning,
the curves generally show an upward trend with the increase
of the number of games. In 10,000 games, the corresponding
cumulative winning is positive, and most of the cumulative win-
ning is positive. However, it should be noted that for the 5-NH
in Fig. 5(a), the curve of 5-players shows a downward trend in
thegamerevenueat1500∼1800gamesand800∼3800games,
and an overall upward trend after 4000 games. The curve of
8-players shows that the cumulative winning also has a down-
ward trend at 3500∼6500 games. In Fig. 5(b), it can be found
that there is a downward trend at 450∼1200 games in the
8-players. In addition, the curve corresponding to the 7-NH also
fluctuates greatly, and the final cumulative winning is relatively
small. Nevertheless, from the curve in Fig. 5, it can be found
that the game performance of Kdb-DeepCFR is better than the
comparison algorithm in the multiplayer game.
The experimental results on the M-NH of Kdb-D2CFR vs.
D2CFRandKdb-D2CFRvs.DREAMareshowninFigs.6(a)and6(b)
respectively.SimilartotheresultsofKdb-DeepCFRinFig.5,from
thecurveofcumulativewinning,theygenerallyshowanupward
trend with the increase of the number of games. From Fig. 6(b),
wecanalsofindthatthecumulativewinningshowsanincreasing
trend when playing against the DREAM. In 10,000 games, the
corresponding cumulative winning is positive, and most of the
cumulative winning in the game is positive. However, it should
be noted that from Fig. 6(a), it can be found that the cumulative
winning of the 8-players is not only negative at 3500∼5000
games,butalsothereisasignificantgapbetweenthecumulative
winningafter10,000gamesandthatinothermultiplayergames.
The corresponding curve of the 7-players also fluctuates greatly,
andthefinalcumulativewinningisrelativelysmall.Nevertheless,
fromthecurveinFig.6,itcanbeseenthatthegameperformance
of kdb-D2CFR is better than the comparison algorithm.
Fig. 7.Game experiment results of Kdb-DeepCFR, Kdb-D2CFR vs. Jaysen on the
6-NH. TheX-axis represents the number of played game, theY-axis represents
the corresponding cumulative winning. The higher the curve, the better the
method.
## Table 1
Game experiment results of Kdb-DeepCFR on the M-NH (mbb/g).
Game player number  3-player4-player5-player
vs. DeepCFR22.1±9.89   17.0±10.99   15.2±12.01
vs. DREAM39.78±11.48  38.10±11.36  25.43±11.66
Game player number  6-player7-player8-player
vs. DeepCFR27.74±12.83  30.02±13.52  17.33±13.37
vs. DREAM11.45±8.92   15.83±10.33  10.39±10.52
Tables 1 and 2 respectively record the average winning of
the proposed algorithm kdb-DeepCFR, kdb-D2CFR and the com-
parison algorithm within the 95% confidence interval for 10,000
games. For kdb-DeepCFR shown in Table 1, its winning when
compared with the comparative algorithm DeepCFR on three to
eight player games are: 22.1±9.89 mbb/g, 17.0±10.99 mbb/g,
## 6

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Fig. 8.Test experiment results of position sensitivity. TheX-axis represents the number of played game, theY-axis represents the corresponding cumulative winning.
The higher the curve, the better the method.
## Table 2
Game experiment results of Kdb-D2CFR on the M-NH (mbb/g).
Game player number  3-player4-player5-player
vs. D2CFR26.02±11.07  15.1±10.66   95.04±14.13
vs. DREAM39.09±10.36  38.34±12.83  25.89±10.33
Game player number  6-player7-player8-player
vs. D2CFR91.08±16.07  6.71±10.75   10.66±12.94
vs. DREAM13.93±10.20  19.42±11.64  12.41±9.06
## Table 3
Test experiment results of position sensitivity (mbb/g).
For AB-B-AB-A-BA-B-B
Game winning   24.44±9.97   35.30±9.813.26±9.58
For BA-A-BA-B-AB-A-A
Game winning−11.4±9.04−27.48±9.42−25.38±10.03
15.2±12.01 mbb/g, 27.74±12.83 mbb/g, 30.02±13.52 mbb/g,
17.33±13.37 mbb/g respectively, and the benefits when com-
pared with the comparative algorithm dream are 39.78±11.48
mbb/g, 38.10±11.36 mbb/g, 25.43±11.66 mbb/g, 11.45±8.92
mbb/g, 15.83±10.33 mbb/g, 10.39±10.52 mbb/g respectively.
For kdb-D2CFR shown in Table 2, its winning when compared
with the comparative algorithm D2CFR on three to eight player
games are: 26.02±11.07 mbb/g, 15.1±10.66 mbb/g, 95.04±
14.13 mbb/g, 91.08±16.07 mbb/g, 6.71±10.75 mbb/g, 10.66±
12.94 mbb/g respectively, and the winning when compared with
the comparative algorithm DREAM are: 39.09±10.36 mbb/g,
## 38.34±12.83mbb/g,25.89±10.33mbb/g,13.93±10.20mbb/g,
19.42±11.64 mbb/g, 12.41±9.06 mbb/g respectively. It can be
found that the proposed kdb-DeepCFR performs better in three
player,sixplayerandsevenplayergamesthanthecomparisonal-
gorithm.Comparedwiththecomparisonalgorithm,theproposed
kdb-D2CFRhasstrongerperformanceinfiveplayerandsixplayer
games,andrelativelypoorperformanceinsevenplayerandeight
player games. From the overall data in the Tables 1 and 2, except
the poor performance of kdb-D2CFR in seven player and eight
person games, the proposed kdb-DeepCFR and kdb-D2CFR have
a leading superiority in other multiplayer games.
Fig. 7 shows the cumulative winning results of Kdb-DeepCFR,
Kdb-D2CFRagainstJaysenonthe6-NHrespectively.Amongthem,
the game mode is one Kdb-DeepCFR and five jaysen agents. The
initial position is random, and the game mode of Kdb-D2CFR is
thesame.ItcanbeseenfromFig.7thatalthoughthecumulative
winning of Kdb-D2CFR at 60∼100 games is a negative return,
and the cumulative winning of Kdb-DeepCFR has a downward
trend at 4000∼6500 games. With the increase of the number
of games, its cumulative winning still shows an upward trend.
And in most cases, it remains a positive winning. This is enough
to show that compared with Jaysen agent, the proposed Kdb-
DeepCFRandKdb-D2CFRhavestrongergameperformanceinthe
game.
After previous analysis and validation analysis of comparative
experiments, in general, compared with previous methods, as
far as we know, these two methods are the first to use neural
networks combined with CFR to effectively solve multiplayer
game strategies, which provides a new idea for the expansion
of the CFR-based method in multiplayer research. Secondly, the
method we proposed effectively realizes the knowledge reuse
from two-player solution to multiplayer solution. And it is also
not involved in previous methods related to CFR-based methods,
and is a good reference for solving multiplayer with CFR-based
methods. In addition, although the proposed method combines
a knowledge distillation framework, it is not complex compared
to the previous six-player agent Pluribus; Finally, our method
can not only effectively solve multiplayer strategies, but also
effectively solve game strategies with 3–8 players, compared to
previousmultiplayersolutionsthatcanonlysolveafixednumber
of game players.
Of course, there are still some areas worth improving in our
method. Firstly, the current strategy accuracy of the proposed
method needs to be improved. Further improvements to the net-
work architecture or more layers and finer network architecture
design may help improve the accuracy of the solution strategy.
In addition, the theoretical guarantee of the proposed method is
lackingcomparedtotheCFRmethod.TheCFRmethodcanensure
that the solution strategy is an approximate Nash equilibrium
strategy in two-player zero-sum games, but it does not have
theoretical guarantees in multiplayer games. The same is true of
the method we proposed. Finally, the proposed method has only
been verified in poker games, especially in Texas Hold’em poker,
and its expansion in other types of applications also requires
further research.
4.2.2. Ablation studies
This section conducts ablation analysis experiments on the
proposed Kdb-DeepCFR and Kdb-D2CFR, mainly from two as-
pects: firstly, this section analyzes the position sensitivity of the
proposed method in multiplayer games, and the experimental
results are shown in Fig. 8 and Table 3; Secondly, this section
analyzes the effectiveness of knowledge distillation in the pro-
posed method. The experimental results are shown in Figs. 9–12
respectively.
Position sensitivity analysis:Inmultiplayergames,especially
in the specific multiplayer game scenario M-NH in the paper, the
finalgameperformancewillbeaffectedbythedifferentpositions
ofthegameplayers.Inordertoconductanexperimentalanalysis
of the proposed method more objectively, this section analyzes
thesensitivityofthegameplayerstotheirpositionsinthegame.
Thetestexperimentiscarriedoutonthe3-NH.OneKdb-DeepCFR
## 7

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Fig. 9.Ablation experiment results on multiplayer games (3–5) of kdb-DeepCFR.
and two DeepCFR, one DeepCFR and two Kdb-DeepCFR are used
for the test game. 10,000 games at each initial position are con-
ducted respectively, and the experimental results are shown in
Fig. 8, where theY-axis represents the cumulative winning of
games, and theX-axis represents the number of played games.
The higher the curve, the better the result. In Fig. 8, ‘‘A’’ refers
to the Kdb-DeepCFR, ‘‘B’’ refers to the comparison algorithm
DeepCFR.‘‘A-B-B’’meansthatKdb-DeepCFRisinthefirstposition
and the two rivals DeepCFR are in other positions. Accordingly,
‘‘B-A-A’’meansDeepCFRisinthefirstposition,andthetworivals
Kdb-DeepCFR are in other positions. In addition, the winning of
the test is the winning of the non-repeated game players among
thethreegameplayers.Thatis,‘‘A-B-B’’isforthewinningof‘‘A’’,
as shown in Fig. 8(a), and ‘‘B-A-A’’ is for the winning of ‘‘B’’, as
shown in Fig. 8(b).
It is obvious from Fig. 8(a) that their winnings are all positive
andgenerallymaintainagradualgrowthtrend.Thatis,whenone
Kdb-DeepCFRisusedtobeagainsttwoDeepCFR,Kdb-DeepCFRis
dominant. It shows that the performance of Kdb-DeepCFR is bet-
ter than DeepCFR, which further verifies the effectiveness of the
proposed method. However, it should be noted that the winning
curvesrepresentingKdb-DeepCFRindifferentgamepositionsare
not the same, and there is a gap between the three curves. This
shows that when Kdb-DeepCFR is in different game positions, its
game winning is different. That is, different game positions will
affect the winning of game players. It can be seen from Fig. 8(a)
that when Kdb-DeepCFR is in the middle position, the winning is
the highest, and when it is in the first place, the winning is the
lowest.
As for the experimental results of ‘‘B-A-A’’ using one DeepCFR
andtwoKdb-DeepCFR,asshowninFig.8(b).Itcanbefoundthat
the winnings of the three curves are negative, and they are gen-
erally in a smaller and smaller trend. This shows that the game
performance of DeepCFR is weaker than that of Kdb-DeepCFR.
Secondly, although the curve ‘‘A-B-A’’ is basically consistent with
the curve ‘‘B-A-A’’, there is still a significant gap between them.
And the curve ‘‘A-A-B’’ further shows that different game posi-
tions will affect the corresponding winnings of game players. It
canbeseenfromFig.8(b)thatwhenthreeplayersplaythegame,
the loss of DeepCFR is the smallest when it is at the end, and the
loss is larger when it is at the first and middle positions.
Table3furtherrecordstherelevantdata.Thetablerecordsthe
average winning of the method within 95% confidence interval
after10,000gamesatdifferentpositions,inunitsofmbb/g.Itcan
be seen that when one Kdb-DeepCFR is used to be against two
DeepCFR, the average winning is 35.30∼9.8 mbb/g when the
position is ‘‘B-A-B’’. While the average winning is 24.44∼9.97
mbb/gand13.26∼9.58mbb/gwhenthepositionis‘‘B-B-A’’and
‘‘A-B-B’’, respectively. When the position is ‘‘B-A-B’’, the winning
is the highest, which also corresponds to the curve in Fig. 8(a).
When using one DeepCFR and two Kdb-DeepCFR, the winnings
at positions ‘‘A-B-A’’ and ‘‘B-A-A’’ are relatively close, which are
−27.48∼9.42 mbb/g and−25.38∼10.03 mbb/g respectively.
When the position is ‘‘A-A-B’’, the winning is−11.4∼9.04
mbb/g, which is relatively small compared with the first two.
## 8

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Fig. 10.Ablation experiment results on multiplayer games (6–8) of kdb-DeepCFR.
## Ingeneral,throughtheexperimentofpositionsensitivityanal-
ysis, the experimental results in Fig. 8 and Table 3 show that
differentgamepositionshaveanimpactonthewinningsofgame
players. In order to make the comparison experiment and other
related experiments more objective and fair, all other experi-
mentscarriedoutinthepaperrandomlyselectthegameposition
of game players, so as to reduce the impact of different game
positions on the winning of game players.
Ablation analysis of knowledge distillation:Knowledge dis-
tillation is an important part of the proposed method. In this
section, ablation analysis experiments are carried out for the
knowledge distillation to verify its effectiveness. The policy net-
work loss and the game performance in the iteration process
are used as the evaluation metrics. Game performance refers
to the average winning of 1000 games with the corresponding
policy network every 50 iterations in the training process. In
order to fully verify the effectiveness of the knowledge distilla-
tion, the experiment was tested on three to eight player games
respectively.
In addition, it should be noted that considering the previous
positionsensitivityanalysis,itisfoundthatnotonlythedifferent
positions will affect the game performance, but also the game
mode selection of ‘‘1 A vs. 2B’’ and ‘‘1B vs. 2A’’ will also affect
the game performance. For this reason, the ‘‘1 A vs. 2B’’ mode
is adopted here. That is, the test method is one, and the com-
parison method is the same method of multiple opponents. The
experimental results are shown in Figs. 9–12.
For the game performance, ‘‘with Kd’’ refers to the proposed
Kdb-DeepCFR with knowledge distillation, ‘‘w/o Kd’’ refers to
Kdb-DeepCFR without knowledge distillation. Their correspond-
ing curves respectively represent the game winnings of playing
against Kdb-DeepCFR when the opponents are all DeepCFR and
the game winnings of playing against DeepCFR when the oppo-
nents are all Kdb-DeepCFR. Therefore, there isno constraint rela-
tionshipbetweenthetwocurveshere.Thecurveonlyreflectsthe
game performance of their respective methods as the only test
method, which can better reflect the impact of knowledge distil-
lation on the method under the same conditions. For policy loss,
it means that the loss of each network changes with the number
of iterations in the corresponding multiplayer game. Figs. 11 and
12showtherelevantablationexperimentsofkdb-D2CFR,andall
settings are consistent with those in Kdb-DeepCFR.
Figs. 9 and 10 show the experimental results of kdb-DeepCFR
after ablation analysis on 3-NH to 5-NH and 6-NH to 8-NH,
respectively. For the game performance, as shown in Figs. 9(a),
9(c), 9(e), 10(a), 10(c) and 10(e). It can be found that the ‘‘with
## Kd’’representedbytheredcurveisbasicallyinapositivewinning
during the game, while the blue dotted line representing ‘‘w/o
Kd’’ is in a negative winning in most cases, which shows the ef-
fectiveness of the knowledge distillation of the proposed method
when playing in the same game mode. However, it should be
noted that first of all, the curve fluctuates greatly, which reflects
thegreatrandomnessoftheexperimentalresults.Thisisbecause
## 9

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Fig. 11.Ablation experiment results on multiplayer games (3–5) of kdb-D2CFR.
thepositionofthemethodissetrandomlyduringthegameinthe
experiment. In the previous position sensitivity analysis experi-
ment,ithasbeenfoundthatdifferentpositionshaveanimpacton
the game winning. However, in order to make the experimental
results more objective, it is still necessary to randomly set the
game position.
For the 3-NH, it can be found from Fig. 9(a) that the game
conducted under different iterations are all positive winnings;
For the 4-NH, it is found from Fig. 9(c) that although its game
winning is slightly less than zero at the 250th iteration, it is still
in the majority of positive winnings on the whole; For the 5-NH,
itsgameperformanceissimilartothatinthe4-NH.FromFig.9(e),
it is found that it is a negative winning in the 200th iteration
and a positive winning in other iterations; For the 6-NH, it is
found from Fig. 10(a) that it is still a positive winning except
for the negative winning in the 100th and 150th iterations; For
the 7-NH, it is found from Fig. 10(c) that the winning is negative
at 50th iteration and 300th iteration; For the 8-NH, it is found
fromFig.10(e)thatthewinningisnegativeat350thiterationand
400th iteration.
Figs. 11 and 12 show the experimental results of kdb-D2CFR
after ablation analysis on 3-NH to 5-NH and 6-NH to 8-NH,
respectively. For the game performance, as shown in Figs. 11(a),
11(c), 11(e), 12(a), 10(c) and 12(e). It means that the red curve
of kdb-D2CFR performs well in the three to six player game,
their game winning is basically positive and has a great leading
advantage. Although the winning in the 300th iteration of the 4-
NH is negative, the winning in other iterations is still positive.
In the 7-NH and the 8-NH, the trend of the game winning is
similartothatofkdb-DeepCFRinthe7-NHandthe8-NH.Onthe
whole,thegameperformanceinthesevenandeightplayergame
is lower than that in the three to six player game.
## Ingeneral,itisobviousthattheknowledgedistillationishelp-
ful for the game performance of the method. However, it should
also be noted that in the 7-NH and 8-NH, the game winning has
been very close to zero, which indicates that the effect of the
methodinthe7-NHand8-NHislimited.Thisisbecausealthough
the knowledge distillation is effective, it has a good performance
in the three to six player game, which shows that the student
model does learn the ‘‘knowledge’’ in the teacher model. How-
ever, this kind of ‘‘knowledge’’ also has its own limitations, and
its effectiveness decreases with the increasing number of game
players.Specifically,inthissectionoftheexperiment,wecanfind
that the game performance of the proposed method on the 7-NH
and 8-NH decreases.
Forthepolicyloss,theresultsareshowninFigs.9(b),9(d),9(f),
10(b),10(d)and10(f).Theyarethecurveofkdb-DeepCFR’spolicy
networklosswithorwithoutknowledgedistillationonthethree
to eight player game with the number of iterations. Figs. 11(b),
11(d), 11(f), 12(b), 10(d) and 12(f) show the corresponding test
curves of kdb-D2CFR on the three to eight player game. It can be
found that no matter in kdb-DeepCFR or kdb-D2CFR, the loss of
thecorrespondingpolicynetworkwith/withoutknowledgedistil-
lation tends to stabilize with the increase of the number of itera-
tions.Itnotonlyshowsthatthepolicynetworkhasconvergedin
## 10

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
Fig. 12.Ablation experiment results on multiplayer games (6–8) of kdb-D2CFR.
thelaterstageoftheiteration,butalsoshowsthattheparameter
setting of the number of iterations in this experiment is reason-
able. The 500 iterations is enough to reflect the quality of the
experimental results in this section. For the red curve ‘‘with kd’’,
itcanbefoundfromthefigurethatcomparedwiththebluecurve
## ‘‘w/okd’’,thecorrespondingpolicynetworkhasachievedconver-
gence with fewer iterations. In addition to the three player game
inFig.11(b),kdb-D2CFRconvergesaroundthe400thiterations.In
othermultiplayergames,bothkdb-DeepCFRandkdb-D2CFRhave
converged at most 200th iterations. In most multiplayer games,
the policy network has converged within 100th iterations. This
fullyverifiestheeffectivenessoftheknowledgedistillationofthe
proposed method.
## 5. Conclusion
In this paper, in order to deal with the problem of IIG with
variant number of players without any expert knowledge and
abstraction technique, we present kdb-DeepCFR and kdb-D2CFR
to solve the strategy of multiplayer IIGs. The methods combine
CFR based methods and knowledge distillation, which can make
the knowledge in two-player IIGs shift to multiplayer IIGs by
knowledge distillation. As far as we know, the kdb-DeepCFR
and kdb-D2CFR are the first DeepCFR based method used to
solve multiplayer IIGs. The extensive experimental results show
that the proposed methods are effective, and outperforms other
comparison methods on multiplayer poker games. In the future,
we will further improve the game performance of this method
through designing network structure and network parameters.
In addition, it is also worth studying to extend this method
to other imperfect-information games. Furthermore, it is also
a very worthwhile work to explore the convergence theory of
equilibrium strategies in multi-player IIGs.
CRediT authorship contribution statement
Huale Li:Conceptualization, Methodology, Data curation,
Writing – original draft.Zengyue Guo:Data curation, Software.
Yang  Liu:Software, Validation.Xuan  Wang:Writing –
reviewing.Shuhan Qi:Writing – review & editing.Jiajia
Zhang:Supervision.Jing Xiao:Validation.
Declaration of competing interest
The authors declare that they have no known competing
financial interests or personal relationships that could have
appeared to influence the work reported in this paper.
Data availability
Data will be made available on request.
## 11

H. Li, Z. Guo, Y. Liu et al.Knowledge-Based Systems 272 (2023) 110567
## Acknowledgments
This research was funded by Ministry of Science and Technol-
ogy of China (2020AAA0104200), Guangdong Provincial Key Lab-
oratory  of  Novel  Security  Intelligence  Technologies
(2022B1212010005), Key Fields Research of Guangdong Province
(2020B0101380001), Shenzhen Foundational Research Funding
(JCYJ20200109113427092, JCYJ20220818102414030), Basic Re-
searchProgramsofTaicang,2022(TC2022JC14),FundamentalRe-
searchFundsfortheCentralUniversities(G2022WD01027,NWPU),
PINGAN-HITsz Intelligence Finance Research Center. The com-
puting resources of Pengcheng Cloud Brain are used in this
research.
## References
[1] D. Fudenberg, D.K. Levine, The Theory of Learning in Games, Vol. 1, Mit
## Press Books, 1998.
[2] R.B. Myerson, Game Theory: Analysis of Conflict, Harvard University Press,
## 1997.
[3] M.J. Osborne, A. Rubinstein, A Course in Game Theory, MIT Press, 1994.
[4] D. Billings, A. Davidson, J. Schaeffer, D. Szafron, The challenge of poker,
## Artificial Intelligence 134 (1–2) (2002) 201–240.
[5] S. McAleer, J.B. Lanier, K.A. Wang, P. Baldi, R. Fox, XDO: A double oracle
algorithm for extensive-form games, Adv. Neural Inf. Process. Syst. 34
## (2021) 23128–23139.
[6] C.-W. Lee, C. Kroer, H. Luo, Last-iterate convergence in extensive-form
games, Adv. Neural Inf. Process. Syst. 34 (2021) 14293–14305.
[7] S. Wang, H. Wang, Q. Gao, L. Hao, Auto-encoder neural network based
prediction of Texas poker opponent’s behavior, Entertain. Comput. 40
## (2022) 100446.
[8] E.Zhao,R.Yan,J.Li,K.Li,J.Xing,AlphaHoldem:High-performanceartificial
intelligence for heads-up no-limit poker via end-to-end reinforcement
learning, in: Proceedings of the AAAI Conference on Artificial Intelligence,
Vol. 36, 2022, pp. 4689–4697.
[9] J. Xu, J. Chen, S. Chen, Efficient opponent exploitation in no-limit Texas
hold’em poker: A neuroevolutionary method combined with reinforcement
learning, Electronics 10 (17) (2021) 2087.
[10] M. Bernasconi-de Luca, F. Cacciamani, S. Fioravanti, N. Gatti, A. Marchesi, F.
Trovò, Exploiting opponents under utility constraints in sequential games,
## Adv. Neural Inf. Process. Syst. 34 (2021) 13177–13188.
[11] M. Bowling, N. Burch, M. Johanson, O. Tammelin, Heads-up limit hold’em
poker is solved, Science 347 (6218) (2015) 145–149.
[12] D. Shi, X. Guo, Y. Liu, W. Fan, Optimal policy of multiplayer poker via
actor-critic reinforcement learning, Entropy 24 (6) (2022) 774.
[13] J. Nash, Non-cooperative games, Ann. of Math. (1951) 286–295.
[14] M. Zinkevich, M. Johanson, M. Bowling, C. Piccione, Regret minimization in
games with incomplete information, in: Advances in Neural Information
Processing Systems, 2008, pp. 1729–1736.
## [15] M. Moravčík, M. Schmid, N. Burch, V. Lis
## `
y, D. Morrill, N. Bard, T.
Davis, K. Waugh, M. Johanson, M. Bowling, Deepstack: Expert-level ar-
tificial intelligence in heads-up no-limit poker, Science 356 (6337) (2017)
## 508–513.
[16] N. Brown, T. Sandholm, Superhuman AI for heads-up no-limit poker:
Libratus beats top professionals, Science 359 (6374) (2017) 1733.
[17] N. Brown, A. Lerer, S. Gross, T. Sandholm, Deep counterfactual regret
minimization, in: International Conference on Machine Learning, 2019, pp.
## 793–802.
[18] H. Li, K. Hu, Z. Ge, T. Jiang, Y. Qi, L. Song, Double neural counterfactual
regret minimization, 2018, arXiv preprint arXiv:1812.10607.
[19] E. Steinberger, Single deep counterfactual regret minimization, 2019, arXiv
preprint arXiv:1901.07621.
[20] H. Li, X. Wang, Z. Guo, J. Zhang, S. Qi, D2CFR: Minimize counterfactual
regret with deep dueling neural network, 2021, arXiv e-prints arXiv–2105.
[21] Y. LeCun, Y. Bengio, G. Hinton, Deep learning, Nature 521 (7553) (2015)
## 436–444.
## [22] C. Tian, M. Zheng, W. Zuo, B. Zhang, Y. Zhang, D. Zhang, Multi-stage
image denoising with the wavelet transform, Pattern Recognit. 134 (2023)
## 109050.
[23] C. Tian, Y. Zhang, W. Zuo, C.-W. Lin, D. Zhang, Y. Yuan, A heterogeneous
group CNN for image super-resolution, IEEE Trans. Neural Netw. Learn.
## Syst. (2022).
[24] H. Li, X. Wang, K. Li, F. Jia, Y. Wu, J. Zhang, S. Qi, Scalable sub-game solving
for imperfect-information games, Knowl.-Based Syst. 231 (2021) 107434.
[25] R. Di Girolamo, C. Esposito, V. Moscato, G. Sperlí, Evolutionary game
theoretical on-line event detection over tweet streams, Knowl.-Based Syst.
## 211 (2021) 106563.
[26] S. Shi, X. Wang, D. Hao, Z. Yang, H. Qu, Solving poker games efficiently:
Adaptive memory based deep counterfactual regret minimization, in: 2022
International Joint Conference on Neural Networks, IJCNN, IEEE, 2022, pp.
## 1–11.
[27] Z. Ge, S. Yang, P. Tian, Z. Chen, Y. Gao, Modeling rationality: Toward better
performance against unknown agents in sequential games, IEEE Trans.
## Cybern. (2022).
[28] P.W. Newall, N. Talberg, Elite professional online poker players: factors
underlying success in a gambling game usually associated with financial
loss and harm, Addict. Res. Theory (2023) 1–12.
[29] X. Xin, Y. Tu, V. Stojanovic, H. Wang, K. Shi, S. He, T. Pan, Online rein-
forcement learning multiplayer non-zero sum games of continuous-time
Markov jump linear systems, Appl. Math. Comput. 412 (2022) 126537.
[30] X. Song, P. Sun, S. Song, V. Stojanovic, Event-driven NN adaptive fixed-time
control for nonlinear systems with guaranteed performance, J. Franklin
## Inst. B 359 (9) (2022) 4138–4159.
[31] K. Arulkumaran, A. Cully, J. Togelius, Alphastar: An evolutionary com-
putation perspective, in: Proceedings of the Genetic and Evolutionary
Computation Conference Companion, 2019, pp. 314–315.
[32] N. Brown, T. Sandholm, Superhuman AI for multiplayer poker, Science 365
## (6456) (2019) 885–890.
[33] A. Rubinstein, Inapproximability of Nash equilibrium, SIAM J. Comput. 47
## (3) (2018) 917–959.
[34] M.J. Osborne, A. Rubinstein, A Course in Game Theory, The MIT Press, 1994.
[35] D.P. Foster, R. Vohra, Regret in the on-line decision problem, Games
## Econom. Behav. 29 (1–2) (1999) 7–35.
[36] R. Gibson, N. Burch, M. Lanctot, D. Szafron, Efficient Monte Carlo counter-
factual regret minimization in games with many player actions, in: Neural
Information Processing Systems, NIPS, 2012.
[37] J.S. Vitter, Random sampling with a reservoir, ACM Trans. Math. Softw. 11
## (1) (1985) 37–57.
[38] G. Hinton, O. Vinyals, J. Dean, Distilling the knowledge in a neural network,
## Comput. Sci. 14 (7) (2015) 38–39.
[39] M.V.Shcherbakov,A.Brebels,N.L.Shcherbakova,A.P.Tyukov,T.A.Janovsky,
V.A. Kamaev, et al., A survey of forecast error measures, World Appl. Sci.
## J. 24 (24) (2013) 171–176.
[40] W. Liu, B. Li, J. Togelius, Model-free neural counterfactual regret
minimization with bootstrap learning, IEEE Trans. Games (2022).
## [41] M. Schmid, N. Burch, M. Lanctot, M. Moravcik, R. Kadlec, M. Bowling,
Variance reduction in monte carlo counterfactual regret minimization (VR-
MCCFR) for extensive form games using baselines, in: Proceedings of the
AAAI Conference on Artificial Intelligence, Vol. 33, 2019, pp. 2157–2164.
[42] G. Farina, C. Kroer, N. Brown, T. Sandholm, Stable-predictive opti-
mistic counterfactual regret minimization, in: International Conference on
Machine Learning, PMLR, 2019, pp. 1853–1862.
[43] G. Farina, C. Kroer, T. Sandholm, Optimistic regret minimization for
extensive-form games via dilated distance-generating functions, Adv.
## Neural Inf. Process. Syst. 32 (2019).
[44] G. Farina, C. Kroer, T. Sandholm, Stochastic regret minimization in
extensive-form games, in: International Conference on Machine Learning,
PMLR, 2020, pp. 3018–3028.
[45] M. Lanctot, E. Lockhart, J.-B. Lespiau, V. Zambaldi, S. Upadhyay, J. Pérolat,
S. Srinivasan, F. Timbers, K. Tuyls, S. Omidshafiei, et al., OpenSpiel: A
framework for reinforcement learning in games, 2019, arXiv preprint
arXiv:1908.09453.
[46] D.P. Kingma, J. Ba, Adam: A method for stochastic optimization, 2014, arXiv
preprint arXiv:1412.6980.
[47] E. Steinberger, A. Lerer, N. Brown, DREAM: Deep Regret minimization with
Advantage baselines and model-free learning, in: Proceedings of the 35th
AAAI Conference on Artificial Intelligence, 2021.
[48] H. Li, X. Wang, S. Qi, Y. Liu, H. Wang, F. Jia, J. Zhang, Solving six-player
games via online situation estimation, in: 2019 IEEE 31st International
Conference on Tools with Artificial Intelligence, ICTAI, IEEE, 2019, pp.
## 1795–1799.
## 12