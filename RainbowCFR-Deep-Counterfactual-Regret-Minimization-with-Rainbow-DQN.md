## RainbowCFR: Deep Counterfactual Regret Minimization with Rainbow DQN

1st Jinquan Liu

School of Computer Science and Technology Harbin University of Science and Technology Harbin, Heilongjiang Province, China 2320400057@stu.hrbust.edu.cn

syzy@hrbust.edu.cn

3rd Xiaodan Wang*

Harbin Far East Institute of Technology Harbin, Heilongjiang Province, China

*wxdhfeit@163.com

success, DeepCFR suffers from high variance in counterfactual value estimation, unstable policy updates, and low sample efficiency, especially in large-scale games [6]–[8].

Abstract—In recent years, incomplete information games (IIGs) have become a central research area in artificial intel- ligence and reinforcement learning. These games, where players lack full knowledge of their opponents’ states, pose signifi- cant challenges to decision-making and policy learning. While Counterfactual Regret Minimization (CFR) and its deep exten- sions, like DeepCFR, have shown promise in solving IIGs by approximating regret values and strategies, they suffer from high variance in counterfactual value estimation, unstable policy updates, and low sample efficiency. To address these challenges, we propose RainbowCFR, a framework that integrates Rainbow DQN’s stabilization mechanisms with CFR. Our method com- bines techniques such as Dueling network architecture, Double Q- learning, C51 (distributed value function estimation), NoisyNet, Prioritized Experience Replay (PER), and N-step returns. Addi- tionally, we introduce a variance-driven Monte Carlo correction module to stabilize counterfactual value estimation and improve learning efficiency. We evaluated RainbowCFR in two poker envi- ronments: Leduc Hold’em and Heads-up No-limit Texas Hold’em (HNLH). Experimental results show that RainbowCFR outper- forms DeepCFR, NFSP, and D2CFR in terms of convergence speed and policy stability, with significant advantages in Leduc Hold’em and robust performance in HNLH, demonstrating the potential of combining Rainbow DQN mechanisms with CFR in IIGs.

To address these challenges, we propose RainbowCFR, an enhanced DeepCFR framework that integrates stabiliza- tion mechanisms from Rainbow DQN with a variance-driven Monte Carlo rectification strategy. RainbowCFR dynamically fuses neural value estimation with Monte Carlo returns to reduce early-stage variance and improve convergence stability [9]–[11].

The main contributions of this work are as follows:

- 1) An improved framework, RainbowCFR, integrating rein- forcement learning stabilization mechanisms and Monte Carlo correction ideas, is proposed for solving games with incomplete information.

- 2) A variance-driven intelligent Monte Carlo correction module is designed to effectively reduce the variance of counterfactual value estimation and improve sample utilization.

- 3) Historical window KL regularization constraints are introduced to achieve the temporal consistency and stability of policy updates.

Index Terms—DeepCFR, Rainbow DQN, Counterfactual Re- gret Minimization, Imperfect Information Games, Reinforcement Learning, Uncertainty Estimation

The algorithm’s performance is verified in the Leduc Hold’em and Heads-up No-limit Texas Hold’em environments using dual evaluation metrics of exploitability and head-to- head performance [12]. The experimental results show that RainbowCFR outperforms existing methods in convergence speed, stability, and game performance.

Imperfect-information games (IIGs) are a fundamental chal- lenge in artificial intelligence and reinforcement learning, where agents must make decisions under uncertainty and partial observability [1]. Such settings arise naturally in strate- gic domains including security, negotiation, and multi-agent decision-making [2], [3].

Our proposed RainbowCFR algorithm is based on the Deep- CFR framework and integrates several improved modules from the Rainbow DQN framework to enhance the stability and uncertainty modeling capabilities of value estimation in Deep- CFR. Through structural optimization and fusion strategies, the algorithm improves training convergence and robustness

Counterfactual Regret Minimization (CFR) and its deep extension, DeepCFR, have become effective approaches for approximating Nash equilibria in IIGs by learning regret and average policies through self-play [4], [5]. Despite their

2nd Yuezhongyi Sun

School of Computer Science and Technology Harbin University of Science and Technology Harbin, Heilongjiang Province, China

## I. INTRODUCTION

## II. METHODS


*Fig. 1. The Architecture of RainbowCFR Algorithm.*

*Fig. 2. Deep Network Model Structure of RainbowCFR.*

while maintaining the theoretical foundation of counterfactual regret minimization.

## A. RainbowCFR Architecture

To address the aforementioned issues, this paper introduces key modules of Rainbow DQN and a variance-driven in- telligent Monte Carlo correction module into the DeepCFR framework, constructing an improved system that integrates deep reinforcement learning and regret minimization ideas, as shown in Fig. 1.

1) Value Network: In the RainbowCFR framework, the value network is responsible for estimating counterfactual value and action advantage at the information set level and is the core of the entire system to achieve stable learning. To improve the expressive power and training stability of the value function, this paper embeds key components of Rainbow DQN into the value network, including the Dueling structure, Double DQN, distributed value function estimation (C51), NoisyNet parameterized noise and the N-step reward [9], as shown in Fig. 2.

2) Policy Network: In RainbowCFR, the Policy Network follows the design of DeepCFR, employing independent mul- tilayer perceptrons (MLPs) to approximate the average pol- icy distribution. This network collects policy samples during tree traversal and performs offline training based on replay samples to minimize the difference between the predicted and target policies. The Policy Network gradually converges to an approximate Nash equilibrium policy through iterative optimization [5].

3) Rectification Module: A major challenge in DeepCFR is the high variance of value estimation during early train- ing stages. To address this issue, RainbowCFR introduces a

variance-driven Monte Carlo rectification module that main- tains statistical estimates (mean and variance) for sampled returns at each information set [13]–[15].

The module records the visit count, sample mean, and variance at each information set. To improve computational efficiency, a variance-driven early stopping mechanism is applied: when the estimated variance falls below a predefined threshold δ and a minimum number of samples is reached, further Monte Carlo sampling is terminated. The variance is defined as:

Where vk is the reward sample k of node I. This mecha- nism can reallocate computational resources to high-variance, information-scarce node regions, thereby improving training efficiency without sacrificing estimation accuracy.

To further encourage exploration in uncertain regions, an intelligent sampling strategy is employed. For each action a, a composite score is computed as:

Score(I, a) Where Q(I, a) = p

represents the estimated action value,

N(I, a) represents the number of visits, T represents the total number of iterations, and λ represents the weight of variance term [14]. Actions with high variance or low visit frequency have higher priority, thus guiding the algorithm to sample in uncertain regions and improving overall sample efficiency [16].

Finally, RainbowCFR employs a dynamic weighted fusion mechanism to combine Monte Carlo estimates VMC(I) and network predictions VNN(I). The dynamic weight w is up- dated based on the iteration count t and a smoothing factor k:

The rectified value V(I) is then computed as V(I) = (1 − α)VNN(I)+αVMC(I). This value is directly used to calculate the immediate counterfactual regrets stored in the advantage memory, ensuring that high-variance neural predictions are corrected by empirical data in the early stages. The operational logic is detailed in Algorithm 1.

4) Policy Network and Loss Function: The policy network follows the standard DeepCFR design and learns the average strategy through supervised learning from historical policy samples. To prevent abrupt policy shifts and strategy forget- ting, a historical-window KL divergence regularization term is introduced [17]:

Where πhist is the average distribution of the policy within the historical window, and γ is the regularization coefficient.


## Algorithm 1 Variance-Driven Correction Logic

Require: Information set I, current variance σ2, iteration t, threshold δ, warmup steps Twarmup

- 1: if σ2 > δ and t < Twarmup then

- 2: Perform MC rollout to obtain return vmc

- 3: Update σ2 using Welford’s online algorithm

- 4: Compute weight w = t t+k

- 5: α = 0.01 + 0.99w

- 6: V(I) = (1 − α)VNN(I) + αvmc

- 7: else

- 8: V(I) = VNN(I)

- 9: end if

10: return V(I)

Ensure:

Corrected value V(I)

## B. Theoretical Discussion: Convergence and Approximation

While tabular CFR has well-established regret bounds (O(1/ √ T)), DeepCFR and its variants, including Rain- bowCFR, rely on function approximation, which complicates strict convergence guarantees. RainbowCFR should be viewed primarily as an empirical enhancement of DeepCFR. Theoreti- cal analysis suggests that convergence to the Nash equilibrium depends heavily on the function approximation error, denoted as ϵapp. RainbowCFR reduces this error through two mecha- nisms: (1) C51 (Distributional RL) captures the multi-modal nature of value distributions in poker more accurately than scalar regression, potentially lowering ϵapp; (2) The Monte Carlo correction module acts as a high-bias, low-variance regularizer in the early training phase, preventing the policy from diverging due to initial neural network errors. Thus, while RainbowCFR is a heuristic method, it tightens the empirical regret bound by minimizing approximation error.

## III. EXPERIMENTS

## A. Experimental Setup

1) Experimental Platform: We evaluate RainbowCFR on two representative two-player zero-sum imperfect-information poker environments: Leduc Hold’em and Heads-up No-limit Texas Hold’em (HNLH) [6], [11], [18]. Leduc Hold’em serves as a controlled benchmark for analyzing convergence speed and stability, while HNLH represents a large-scale, high- dimensional game that tests robustness and generalization [19].

All experiments are conducted using the OpenSpiel frame- work [20]. Baseline methods include NFSP, DeepCFR, and D2CFR, implemented using official or publicly available con- figurations [21]. For RainbowCFR, both the value network and policy network use multilayer perceptrons with seven fully connected layers. Training is performed using the AdamW

optimizer with a learning rate of 1 × 10−3. The replay buffer capacity is set to 106, and prioritized experience replay is enabled. Each experiment runs for 1000 CFR iterations with 800 external sampling traversals per iteration.

Algorithm performance is evaluated using exploitability and head-to-head performance. Lower exploitability indicates closer approximation to the Nash equilibrium. In HNLH, final

*Fig. 3. The Y-axis denotes exploitability and the X-axis denotes the number of iterations. The lower the value, the better.*

strategies are further assessed through 10,000 head-to-head games against baseline agents.

## B. Experimental Results

We designed two sets of experiments to fully verify the performance of the proposed method: the first set of experi- ments was conducted on the heads-up no-limit Texas Hold’em (HNLH) platform, comparing RainbowCFR with the current mainstream advanced methods (NFSP, DeepCFR, D2CFR) to directly verify its overall effectiveness; the second set of experiments was an ablation experiment, which aimed to quantify the independent contribution of each improved component of RainbowCFR to the performance.

1) Comparison with state-of-the-art methods: The first set of experiments selected three main advanced methods from recent years as baseline comparisons: NFSP, DeepCFR, and D2CFR. To fully verify the performance advantages of the proposed method, firstly, its exploitability was compared with DeepCFR, D2CFR, and NFSP on the Leduc platform, and the results are shown in Fig. 3. Secondly, pairwise comparison experiments were conducted on two test platforms: Leduc and HNLH, to quantitatively evaluate its competitive performance, and the relevant results are shown in Table I.

Fig.3 shows that RainbowCFR consistently maintains the best exploitability performance throughout the training pro- cess. Specifically, in the early stages of training, Rain- bowCFR’s rate of decline is significantly faster than other methods, and it stabilizes in the lead after approximately 100 iterations. In the mid-to-late stages, the RainbowCFR’s curve maintains a steady downward trend and remains at the lowest level among all algorithms, without significant fluctuations or performance degradation.

In addition to these standard baselines, we also consider recent approaches such as SD-CFR [22] and DREAM [23]. While SD-CFR attempts to remove the need for a separate value network by directly learning the policy, our exper- iments suggest that the distributional perspective (C51) in RainbowCFR offers superior stability in high-variance scenar- ios. Similarly, while DREAM achieves low variance through learned baselines, RainbowCFR achieves comparable variance reduction via the explicit Monte Carlo correction module without the overhead of an additional baseline network.


*TABLE I*

*HEAD-TO-HEAD PERFORMANCE OF RAINBOWCFR*

| Tested game | DCFR | NFSP | D2CFR |
| --- | --- | --- | --- |
| Leduc RainbowCFR vs |   | 300.2 ± 48.5 260.7 ± 39.4 35.8 ± 9.7 |   |
| HNLH RainbowCFR vs |   | 90.4 ± 17.9 130.3 ± 26.1 20.6 ± 7.9 |   |

*TABLE II*

*INCREMENTAL ABLATION OF RAINBOW COMPONENTS ON LEDUC*

*HOLD’EM (EXPLOITABILITY MBB/G)*

| Model Configuration | Iter 100 (Speed) | Iter 1000 (Final) |
| --- | --- | --- |
| Base DeepCFR | 450.2 | 120.5 |
| + PER | 380.1 | 115.3 |
| + PER + N-step | 365.4 | 110.2 |
| + PER + N-step + C51 | 300.2 | 45.8 |

Table I presents RainbowCFR’s head-to-head performance in Leduc Hold’em and HNLH environments, with results presented as milli-big blinds per game (mbb/g). The table reports the average payoff and its 95% confidence interval.

In the Leduc Hold’em environment, RainbowCFR out- performed DeepCFR, NFSP, and D2CFR by 300.2±48.5, 260.7±39.4, and 35.8±9.7 mbb/g respectively, demonstrating significant strategic stability and adversarial ability in medium- scale information games.

In the more complex HNLH environment, RainbowCFR achieved average performance advantages of 90.4±17.9 mbb/g over DeepCFR, 130.3±26.1 mbb/g over NFSP, and 20.6±7.9 mbb/g over D2CFR. Although the overall return was lower than in the Leduc environment, it still maintained stable positive returns, demonstrating the algorithm’s robustness and generalization performance in large-scale high-dimensional games.

2) Ablation Study: In this section, we conduct a systematic ablation study to disentangle the contributions of individual Rainbow components. Unlike previous global ablations, here we incrementally add Prioritized Experience Replay (PER), N- step Returns, and C51 Distributional RL to the base DeepCFR model.

Table II summarizes the isolated effects of these compo- nents. The results indicate that PER contributes most sig- nificantly to sample efficiency in the early stages (0-300 iterations). C51 is the primary driver for reducing exploitability variance in the later stages (final convergence), as it better captures the multi-modal value distribution inherent in poker. NoisyNet provides marginal improvement in Leduc Hold’em but is essential for exploration in the larger HNLH environ- ment.

Fig. 4 evaluates the effect of removing Monte Carlo rectifi- cation and dynamic value fusion. Without Monte Carlo guid- ance, convergence becomes slower and less stable, confirming its critical role in variance reduction.

Fig. 5 illustrates the impact of prioritized experience replay. Removing PER results in increased oscillations and slower convergence, indicating that prioritized sampling improves learning efficiency and stability.

Overall, the experimental results confirm that combining Rainbow-based stabilization with variance-driven Monte Carlo

*Fig. 4. Ablation Study on Dynamic Fusion of Monte Carlo and Neural Network Value Estimation in RainbowCFR.*

*Fig. 5. Ablation Study on Prioritized Experience Replay in RainbowCFR.*

rectification significantly improves convergence stability and robustness in imperfect-information games.

## IV. CONCLUSION AND FUTURE WORK

In this paper, we presented RainbowCFR, an enhanced DeepCFR algorithm that integrates the Rainbow DQN frame- work to address uncertainty estimation and stability issues in imperfect-information games. Through the incorporation of prioritized replay, double Q-learning, noisy exploration, and distributional value representation, RainbowCFR achieves substantial improvements in both convergence speed and ro- bustness.

The proposed confidence-weighted Monte Carlo fusion effectively balances empirical sampling and neural predic- tion, leading to lower exploitability and smoother training dynamics. Our experiments demonstrate that RainbowCFR outperforms DeepCFR, NFSP, and D2CFR.

In future work, we plan to extend RainbowCFR to:

- 1) Continuous and multi-agent environments, exploring scalability to large action spaces [24];

- 2) Hierarchical or meta-CFR frameworks, integrating multi-level reasoning [25];

- 3) Uncertainty quantification in real-world decision- making, such as negotiation and autonomous control [26].

This study demonstrates that integrating advanced reinforce- ment learning techniques into regret-minimization frameworks is a promising direction for building stable, uncertainty-aware multi-agent systems.


## REFERENCES

- [1] N. Brown, A. Bakhtin, A. Lerer, Q. Gong, et al., “Combining deep reinforcement learning and search for imperfect-information games,” in Advances in Neural Information Processing Systems, vol. 33, pp. 17057–17069, 2020.

- [2] J. Heinrich, “Reinforcement learning from self-play in imperfect- information games,” Ph.D. dissertation, Univ. College London, London, UK, 2017.

- [3] M. Liu, G. Farina, and A. Ozdaglar, “A policy-gradient approach to solving imperfect-information games with iterate convergence,” arXiv preprint arXiv:2408.00751, 2024.

- [4] M. Zinkevich, M. Johanson, M. Bowling, et al., “Regret minimization in games with incomplete information,” in Advances in Neural Information Processing Systems, vol. 20, 2007.

- [5] N. Brown, A. Lerer, S. Gross, et al., “Deep counterfactual regret minimization,” in Proc. Int. Conf. Machine Learning, PMLR, 2019, pp. 793–802.

- [6] M. Bowling, N. Burch, M. Johanson, et al., “Heads-up limit hold’em poker is solved,” Science, vol. 347, no. 6218, pp. 145–149, 2015.

- [7] M. Lanctot, K. Waugh, M. Zinkevich, et al., “Monte Carlo sampling for regret minimization in extensive games,” in Advances in Neural Information Processing Systems, vol. 22, 2009.

- [8] M. E. Lorasdagi, D. C. Cicek, F. B. Mutlu, et al., “Enhancing deep de- terministic policy gradients on continuous control tasks with decoupled prioritized experience replay,” arXiv preprint arXiv:2512.05320, 2025.

- [9] M. Hessel, J. Modayil, H. van Hasselt, et al., “Rainbow: Combining improvements in deep reinforcement learning,” in Proc. AAAI Conf. Artificial Intelligence, vol. 32, no. 1, 2018.

- [10] J. K. V. Kumar and V. K. Elumalai, “A proximal policy optimization based deep reinforcement learning framework for tracking control of a flexible robotic manipulator,” Engineering Science and Technology, Int. J., 2025.

- [11] N. Brown and T. Sandholm, “Superhuman AI for multiplayer poker,” Science, vol. 365, no. 6456, pp. 885–890, 2019.

- [12] H. Li, X. Qian, and W. Song, “Prioritized experience replay based on dynamics priority,” Scientific Reports, vol. 14, no. 1, p. 6014, 2024.

- [13] H. Li, X. Wang, Z. Guo, et al., “D2CFR: Minimize counterfactual regret with deep dueling neural network,” IEEE Trans. Neural Netw. Learn. Syst., 2023.

- [14] H. Xu, K. Li, B. Liu, et al., “Minimizing weighted counterfactual regret with optimistic online mirror descent,” arXiv preprint arXiv:2404.13891, 2024.

- [15] J. Chen, T. Lan, and V. Aggarwal, “Hierarchical deep counterfactual regret minimization,” arXiv preprint arXiv:2305.17327, 2023.

- [16] B. Li, Z. Fang, and L. Huang, “RL-CFR: Improving action abstraction for imperfect information extensive-form games with reinforcement learning,” in Proc. 41st Int. Conf. Machine Learning, Proc. Mach. Learn. Res., vol. 235, pp. 27752–27770, 2024.

- [17] J. Queeney, I. Ch. Paschalidis, and C. G. Cassandras, “Generalized policy improvement algorithms with theoretically supported sample reuse,” IEEE Trans. Autom. Control, 2025.

- [18] T. S. Ferguson, A Course in Game Theory. Singapore: World Scientific, 2020.

- [19] N. Brown and T. Sandholm, “Safe and nested subgame solving for imperfect-information games,” in Advances in Neural Information Pro- cessing Systems, vol. 30, pp. 690–700, 2017.

- [20] M. Lanctot, E. Lockhart, J. B. Lespiau, et al., “OpenSpiel: A framework for reinforcement learning in games,” arXiv preprint arXiv:1908.09453, 2019.

- [21] H. Zhang, Y. Lei, L. Gui, et al., “CPPO: Continual learning for reinforcement learning with human feedback,” in Proc. Int. Conf. Learn. Representations, 2024.

- [22] E. Steinberger, ”Single deep counterfactual regret minimization,” arXiv preprint arXiv:1901.07621, 2019.

- [23] E. Steinberger, A. Lerer, and N. Brown, ”DREAM: Deep regret min- imization with advantage baselines and model-free learning,” in Proc. Int. Conf. Learn. Representations, 2020.

- [24] Z. Ning and L. Xie, “A survey on multi-agent reinforcement learning and its application,” J. Artif. Intell., vol. 2, no. 4, pp. 1–28, 2024.

- [25] K. Li, H. Xu, H. Fu, Q. Fu, and J. Xing, “Automatically designing counterfactual regret minimization algorithms for solving imperfect- information games,” Artificial Intelligence, vol. 337, 104232, Dec. 2024.

- [26] Y. Shi, P. Wei, K. Feng, D.-C. Feng, and M. Beer, “A survey on machine learning approaches for uncertainty quantification of engineering sys- tems,” Machine Learning for Computational Science and Engineering, vol. 1, art. 11, 2025.
