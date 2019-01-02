:cat:[Facial Expression Recognition (FER)](#facial-expression-recognition)  

:cat:[Cross-Dataset FER](#cross-dataset-fer)  

:cat:[Differences in FER](#differences-in-fer)  

:cat:[FER Datasets](#fer-datasets)

:cat:[FER Challenges](#fer-challenges)

:cat:[Affective Level Estimation](#affective-level-estimation)

:cat:[Common Domain Adaptation](#common-domain-adaptation)

:cat:[Other Cross-Domain Tasks](#other-cross-domain-tasks)

:cat:[Image to Video Adaptation](#image-to-video-adaptation)

:cat:[Attention](#attention)

## Facial Expression Recognition

### AAAI19
- CycleEmotionGAN: Emotional Semantic Consistency Preserved CycleGAN for Adapting
Image Emotions [[paper]()]

### ACM MM18
- Fast and Light Manifold CNN based 3D Facial Expression Recognition across Pose Variations [[paper](https://dl.acm.org/citation.cfm?id=3240568)]
- Facial Expression Recognition in the Wild: A Cycle-Consistent Adversarial Attention Transfer Approach [[paper](https://dl.acm.org/citation.cfm?id=3240574)]
- Facial Expression Recognition Enhanced by Thermal Images through Adversarial Learning [[paper](https://dl.acm.org/citation.cfm?id=3240608)]
- Geometry Guided Adversarial Facial Expression Synthesis [[paper](https://arxiv.org/pdf/1712.03474.pdf)]
- Conditional Expression Synthesis with Face Parsing Transformation [[paper](https://dl.acm.org/citation.cfm?id=3240647)]

### ECCV18 
- Facial Expression Recognition with Inconsistently Annotated Datasets [[paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Jiabei_Zeng_Facial_Expression_Recognition_ECCV_2018_paper.pdf)]
- Contemplating Visual Emotions: Understanding and Overcoming Dataset Bias [[paper](https://arxiv.org/pdf/1808.02212.pdf)]
- Deep Multi-Task Learning to Recognise Subtle
Facial Expressions of Mental States [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guosheng_Hu_Deep_Multi-Task_Learning_ECCV_2018_paper.pdf)]

### ICML18 
- Video Prediction with Appearance and Motion Conditions [[paper](https://arxiv.org/pdf/1807.02635.pdf)],[[project](https://sites.google.com/vision.snu.ac.kr/icml2018-video-prediction)][:dizzy::dizzy:]

### CVPR18
- Facial Expression Recognition by De-expression Residue Learning [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_Facial_Expression_Recognition_CVPR_2018_paper.pdf)][:dizzy::dizzy::dizzy:]
- Joint Pose and Expression Modeling for Facial Expression Recognition [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Joint_Pose_and_CVPR_2018_paper.pdf)],[[code](https://github.com/FFZhang1231/Facial-expression-recognition)][:dizzy::dizzy:]
- 4DFAB: A Large Scale 4D Database for Facial Expression Analysis and Biometric Applications [[paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/3299.pdf)][:dizzy:]
- (workshop) Covariance Pooling for Facial Expression Recognition [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w10/Acharya_Covariance_Pooling_for_CVPR_2018_paper.pdf)],[[code](https://github.com/d-acharya/CovPoolFER)][:dizzy:]
- (workshop) Unsupervised Features for Facial Expression Intensity Estimation over Time [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w17/Awiszus_Unsupervised_Features_for_CVPR_2018_paper.pdf)][:dizzy::dizzy:]
- (workshop) A Compact Deep Learning Model for Robust Facial Expression Recognition [[paper](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w41/Kuo_A_Compact_Deep_CVPR_2018_paper.pdf)][:dizzy:]

### AAAI18
-  ExprGAN: Facial Expression Editing with Controllable Expression Intensity [[paper](https://arxiv.org/pdf/1709.03842.pdf)],[[code](https://github.com/HuiDingUMD/ExprGAN)][:dizzy:]
-  Learning Spatio-temporal Features with Partial Expression Sequences for
on-the-Fly Prediction [[paper](https://arxiv.org/pdf/1711.10914.pdf)][:dizzy:]

### IJCAI18
- Personality-Aware Personalized Emotion Recognition from Physiological Signals [[paper](https://www.ijcai.org/proceedings/2018/0230.pdf)]

### FG18 (Access Provided by Authenticated Institutes)
- Multi-Channel Pose-Aware Convolution Neural Networks for Multi-View Facial Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373867)][:dizzy:]
- Automatic 4D Facial Expression Recognition using Dynamic
Geometrical Image Network [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373807)][:dizzy:]
- ExpNet: Landmark-Free, Deep, 3D Facial Expressions [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373820)][:dizzy:]
- Perceptual Facial Expression Representation [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373828)][:dizzy:]
- Emotion-Preserving Representation Learning via Generative Adversarial Network
for Multi-view Facial Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373839)][:dizzy::dizzy:]
- Spotting the Details: The Various Facets of Facial Expressions [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373842)][:dizzy:]
- Identity-Adaptive Facial Expression Recognition Through Expression Regeneration Using Conditional Generative Adversarial Networks [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373843)][:dizzy::dizzy::dizzy:]
- Hand-crafted Feature Guided Deep Learning for Facial Expression
Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373861)][:dizzy::dizzy:]
- Accurate Facial Parts Localization and Deep Learning for 3D Facial Expression
Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373868)][:dizzy:]
- Changes in Facial Expression as Biometric: A
Database and Benchmarks of Identification [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373891)][:dizzy:]
- LTP-ML : Micro-Expression Detection by Recognition of Local temporal Pattern of Facial Movements [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373893)][:dizzy:]
- From Macro to Micro Expression Recognition: Deep Learning on Small Datasets
Using Transfer Learning  [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373896)][:dizzy:]

### IEEE Trans on Affective Computing18 (Access Provided by Authenticated Institutes)
- Deep Learning for Human Affect Recognition: Insights and New Developments [[paper](https://ieeexplore.ieee.org/document/8598999)]
- Facial Expression Recognition with Identity and Emotion Joint Learning [[paper](https://ieeexplore.ieee.org/document/8528894)]
- Unsupervised adaptation of a person-specific
manifold of facial expressions [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8294217)][:dizzy::dizzy::dizzy:]
- Multi-velocity neural networks for facial
expression recognition in videos [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7942120)][:dizzy::dizzy::dizzy:]
- Multi-Objective based Spatio-Temporal
Feature Representation Learning Robust to
Expression Intensity Variations for Facial
Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7904596)]
- Visually Interpretable Representation Learning for
Depression Recognition from Facial Images [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8344107)][:dizzy::dizzy::dizzy::dizzy:]
- An Adaptive Bayesian Source Separation
Method for Intensity Estimation of Facial AUs [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7933209)][:dizzy::dizzy::dizzy::dizzy:]
- Facial Expression Recognition in Video
with Multiple Feature Fusion [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7518582)][:dizzy::dizzy:]

### CVPR17
- Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression
Recognition in the Wild [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Reliable_Crowdsourcing_and_CVPR_2017_paper.pdf)][:dizzy:]
- Emotion Recognition in Context [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kosti_Emotion_Recognition_in_CVPR_2017_paper.pdf)][:dizzy:]
- (workshop) Facial Expression Recognition Using Enhanced Deep 3D Convolutional Neural Networks [[paper](https://arxiv.org/pdf/1705.07871.pdf)][:dizzy:]
- (workshop) Estimation of Affective Level in the Wild
With Multiple Memory Networks [[paper](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Li_Estimation_of_Affective_CVPR_2017_paper.pdf)][:dizzy::dizzy:]
- (workshop) DyadGAN: Generating Facial Expressions in Dyadic Interactions [[paper](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w41/papers/Khan_DyadGAN_Generating_Facial_CVPR_2017_paper.pdf)][:dizzy:]
- (workshop) Personalized Automatic Estimation of Self-reported Pain Intensity
from Facial Expressions [[paper](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w41/papers/Picard_Personalized_Automatic_Estimation_CVPR_2017_paper.pdf)][:dizzy::dizzy:]

### ICCV17
- A Novel Space-Time Representation on the Positive Semidefinite Cone
for Facial Expression Recognition [[paper](https://arxiv.org/pdf/1707.06440.pdf)][:dizzy:]
- (workshop) Facial Expression Recognition via Joint Deep Learning of RGB-Depth Map
Latent Representations [[paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w44/Oyedotun_Facial_Expression_Recognition_ICCV_2017_paper.pdf)][:dizzy:]
- (workshop) Facial Expression Recognition using Visual Saliency and Deep Learning [[paper](https://arxiv.org/pdf/1708.08016.pdf)][:dizzy::dizzy:]

### FG17
- Accurate Facial Parts Localization and Deep Learning for 3D Facial
Expression Recognition [[paper](https://arxiv.org/pdf/1803.05846.pdf)][:dizzy:]
- FaceNet2ExpNet: Regularizing a Deep Face Recognition Net for
Expression Recognition [[paper](https://arxiv.org/pdf/1609.06591.pdf)][:dizzy::dizzy:]
- Deep generative-contrastive networks for facial expression recognition [[paper](https://arxiv.org/pdf/1703.07140.pdf)][:dizzy::dizzy::dizzy:]
- Identity-Aware Convolutional Neural Network for Facial Expression
Recognition [[paper](https://cse.sc.edu/~mengz/papers/FG2017.pdf)][:dizzy::dizzy::dizzy:]
- (workshop) Spatio-Temporal Facial Expression Recognition Using Convolutional
Neural Networks and Conditional Random Fields [[paper](https://arxiv.org/pdf/1703.06995.pdf)][:dizzy:]
- Head Pose and Expression Transfer using Facial Status Score [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7961793)][:dizzy:]
- Sayette Group Formation Task (GFT)
Spontaneous Facial Expression Database [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7961794)][:dizzy::dizzy:]
- Curriculum Learning for Facial Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7961783)][:dizzy:]
- Implicit Media Tagging and Affect Prediction from RGB-D video of
spontaneous facial expressions [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7961813)][:dizzy::dizzy:]


### CVPR16
- LOMo: Latent Ordinal Model for Facial Analysis in Videos [[paper](http://www.grvsharma.com/hpresources/lomo_cvpr16_arxiv.pdf)][:dizzy:]
- Facial Expression Intensity Estimation Using Ordinal Information [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhao_Facial_Expression_Intensity_CVPR_2016_paper.pdf)],[[Supplementary](http://openaccess.thecvf.com/content_cvpr_2016/supplemental/Zhao_Facial_Expression_Intensity_2016_CVPR_supplemental.pdf)][:dizzy::dizzy:]
- EmotioNet: An accurate, real-time algorithm for the automatic annotation of a
million facial expressions in the wild [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Benitez-Quiroz_EmotioNet_An_Accurate_CVPR_2016_paper.pdf)],[[Supplementary](http://openaccess.thecvf.com/content_cvpr_2016/supplemental/Benitez-Quiroz_EmotioNet_An_Accurate_2016_CVPR_supplemental.pdf)][:dizzy:]
- Multimodal Spontaneous Emotion Corpus for Human Behavior Analysis [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhang_Multimodal_Spontaneous_Emotion_CVPR_2016_paper.pdf)][:dizzy:]
- (workshop) Automatic Recognition of Emotions and Membership in Group Videos [[paper](http://openaccess.thecvf.com/content_cvpr_2016_workshops/w28/papers/Mou_Automatic_Recognition_of_CVPR_2016_paper.pdf)][:dizzy:]
- (workshop) Extended DISFA Dataset: Investigating Posed and Spontaneous Facial
Expressions [[paper](http://openaccess.thecvf.com/content_cvpr_2016_workshops/w28/papers/Mavadati_Extended_DISFA_Dataset_CVPR_2016_paper.pdf)][:dizzy:]

### ECCV16
- Peak-Piloted Deep Network for Facial Expression
Recognition [[paper](https://arxiv.org/pdf/1607.06997.pdf)][:dizzy::dizzy::dizzy:]

### WACV16
- Going Deeper in Facial Expression Recognition using Deep Neural Networks [[paper](https://arxiv.org/pdf/1511.04110.pdf)][:dizzy:]

### ICCV15
- Joint Fine-Tuning in Deep Neural Networks
for Facial Expression Recognition [[paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Jung_Joint_Fine-Tuning_in_ICCV_2015_paper.pdf)][:dizzy::dizzy::dizzy:]
- Pairwise Conditional Random Forests for Facial Expression Recognition [[paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Dapogny_Pairwise_Conditional_Random_ICCV_2015_paper.pdf)][:dizzy:]

### FG15
- Pairwise Linear Regression: An Efficient and Fast Multi-view Facial
Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7163101)][:dizzy:]

### CVPR14
- Learning Expressionlets on Spatio-Temporal Manifold for Dynamic Facial
Expression Recognition [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Liu_Learning_Expressionlets_on_2014_CVPR_paper.pdf)][:dizzy::dizzy:]
- Facial Expression Recognition via a Boosted Deep Belief Network [[paper](http://openaccess.thecvf.com/content_cvpr_2014/papers/Liu_Facial_Expression_Recognition_2014_CVPR_paper.pdf)][:dizzy:]

### CVPR13
- Capturing Complex Spatio-Temporal Relations among Facial
Muscles for Facial Expression Recognition [[paper](http://f4k.dieei.unict.it/proceedings/CVPR13/data/papers/4989d422.pdf)][:dizzy:]

### Others
- (IEEE Transactions on Image Processing18) Occlusion aware facial expression recognition using
CNN with attention mechanism [[paper](https://ieeexplore.ieee.org/document/8576656)]
- Visual Saliency Maps Can Apply to Facial Expression Recognition [[paper](https://arxiv.org/pdf/1811.04544.pdf)]
- (IEEE Access18) SMEConvNet: A Convolutional Neural Network
for Spotting Spontaneous Facial
Micro-Expression From Long Videos [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8522030)]
- (Google AI) A Compact Embedding for Facial Expression Similarity [[paper](https://arxiv.org/pdf/1811.11283.pdf)]
- (submission to IJCAI-ECAI 2018) Geometry-Contrastive Generative Adversarial Network for Facial Expression Synthesis [[paper](https://arxiv.org/abs/1802.01822)][:dizzy::dizzy::dizzy:]
-  Deep Facial Expression Recognition: A Survey [[paper](https://arxiv.org/pdf/1804.08348.pdf)][:dizzy::dizzy:]
-  (ACM Computing Surveys) Facial Expression Analysis under Partial Occlusion: A Survey [[paper](https://arxiv.org/pdf/1802.08784.pdf)][:dizzy:]
-  Expression Empowered ResiDen Network for Facial Action Unit Detection [[paper](https://arxiv.org/pdf/1806.04957.pdf)][:dizzy:]
-  Deep Covariance Descriptors for Facial
Expression Recognition [[paper](https://arxiv.org/pdf/1805.03869.pdf)][:dizzy:]
- Fine-Grained Facial Expression Analysis Using
Dimensional Emotion Model [[paper](https://arxiv.org/pdf/1805.01024.pdf)][:dizzy:]
- VGAN-Based Image Representation Learning
for Privacy-Preserving Facial Expression Recognition [[paper](https://arxiv.org/pdf/1803.07100v1.pdf)][:dizzy:]
- Non-Volume Preserving-based Feature Fusion Approach
to Group-Level Expression Recognition on Crowd Videos [[paper](https://arxiv.org/pdf/1811.11849.pdf)]
- (MIT16) Predicting Perceived Emotions in Animated GIFs
with 3D Convolutional Neural Networks [[paper](https://affect.media.mit.edu/pdfs/16.Chen-etal-ISM.pdf)]
- (MIT CSAIL18) Controllable Image-to-Video Translation:
A Case Study on Facial Expression Generation [[paper](https://arxiv.org/pdf/1808.02992.pdf)] 

## Cross-Dataset FER 
- Cross-database non-frontal facial expression
recognition based on transductive deep transfer
learning [[paper](https://arxiv.org/pdf/1811.12774.pdf)]
- Unsupervised Domain Adaptation for Facial Expression Recognition Using Generative Adversarial Networks [[paper](https://www.hindawi.com/journals/cin/2018/7208794/)]
- ICPR 18 Deep Emotion Transfer Network for Cross-database Facial Expression Recognition 
- FG 2018 Deep Unsupervised Domain Adaptation for Face Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373866)][:dizzy::dizzy::dizzy:]
- FG 2018 Cross-generating GAN for Facial Identity Preserving [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373821)][:dizzy:]
- ACM MM 2017 Learning a Target Sample Re-Generator for Cross-Database
Micro-Expression Recognition [[paper](https://arxiv.org/pdf/1707.08645.pdf)][:dizzy:]
- TIP 2018 Domain Regeneration for Cross-Database
Micro-Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8268553)][:dizzy:]
- CVPR 2013 Selective Transfer Machine for Personalized Facial Action Unit Detection [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=6619295)][:dizzy:]
- TPAMI 2017 Selective Transfer Machine for Personalized
Facial Expression Analysis [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7442563)]
- ICONIP 2016 Cross-Database Facial Expression Recognition
via Unsupervised Domain Adaptive
Dictionary Learning [[paper](https://link-springer-com.eproxy1.lib.hku.hk/content/pdf/10.1007%2F978-3-319-46672-9.pdf)](P428)[:dizzy:]
- IEEE TRANSACTIONS ON AFFECTIVE COMPUTING 2018 Cross-Domain Color Facial Expression Recognition Using Transductive Transfer
Subspace Learning [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7465718)][:dizzy:]
- FG2018 Unsupervised Domain Adaptation with Regularized Optimal Transport
for Multimodal 2D+3D Facial Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373808)][:dizzy::dizzy:]
- ICB2016 Discriminative Feature Adaptation
for Cross-Domain Facial Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7550085)][:dizzy:]
- ICB2015 A Transfer Learning Approach to Cross-Database Facial Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7139098)][:dizzy:]
- ICRA2011 Cross-dataset facial expression recognition [[paper](https://ieeexplore.ieee.org/document/5979705/)][:dizzy:]
- Neurocomputing2016 Transfer subspace learning for cross-dataset facial expression recognition [[paper](https://ac-els-cdn-com.eproxy1.lib.hku.hk/S0925231216304623/1-s2.0-S0925231216304623-main.pdf?_tid=dd93f446-9503-481b-9fd4-0a8011ac55cc&acdnat=1531015927_ff7fb7e1b50c81933244378c57033099)][:dizzy:]
- IEEE TRANSACTIONS ON MULTIMEDIA 2016 A Deep Neural Network-Driven Feature Learning Method for Multi-view
Facial Expression Recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7530823)](Cross-view facial expression
recognition)[:dizzy:]
- CVPR2014 Constrained Deep Transfer Feature Learning and its Applications [[paper](https://arxiv.org/pdf/1709.08128.pdf)](Cross-view facial expression
recognition)[:dizzy:]
- IEEE SIGNAL PROCESSING LETTERS 2016 Cross-Corpus Speech Emotion Recognition Based on Domain-Adaptive Least-Squares Regression [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7425198)][:dizzy:]
- ICMI2017 Cross-Modality Interaction between EEG Signals and Facial
Expression [[paper](http://delivery.acm.org.eproxy1.lib.hku.hk/10.1145/3140000/3137034/icmi17-dc-111.pdf?ip=147.8.31.43&id=3137034&acc=ACTIVE%20SERVICE&key=CDD1E79C27AC4E65%2EDE0A32330AE3471B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1530995113_e379f7ad86d3c2ead3f959f909b4d496)][:dizzy:]
- ACM MM 2017 Integrated Face Analytics Networks through
Cross-Dataset Hybrid Training[[paper](http://delivery.acm.org.eproxy1.lib.hku.hk/10.1145/3130000/3123438/p1531-li.pdf?ip=147.8.31.43&id=3123438&acc=ACTIVE%20SERVICE&key=CDD1E79C27AC4E65%2EDE0A32330AE3471B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1530997759_33591edce0f41f32f5c70d5902b2dd01#URLTOKEN#)][:dizzy::dizzy:]

## Differences in FER
- Differences in Facial Emotional Recognition Between Patients With the First-Episode Psychosis, Multi-episode Schizophrenia, and Healthy Controls [[paper](https://www.cambridge.org/core/journals/journal-of-the-international-neuropsychological-society/article/differences-in-facial-emotional-recognition-between-patients-with-the-firstepisode-psychosis-multiepisode-schizophrenia-and-healthy-controls/08C019C4C4E210BFBBDEA950031BA5E3)]

## FER Datasets
- [Jaffe](http://www.kasrl.org/jaffe.html)
- [FER-2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- [MMI Facial Expression Database](https://www.mmifacedb.eu/)
- [Cohn-Kanade Expression Database](http://www.pitt.edu/~emotion/ck-spread.htm)
- [Oulu-CASIA NIR&VIS facial expression database](http://www.cse.oulu.fi/CMV/Downloads/Oulu-CASIA)
- [Multi-PIE](http://www.flintbox.com/public/project/4742/)
- [BU-3DFE (Binghamton University 3D Facial Expression) Database](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)
- [Real-world Affective Faces (RAF) Database](http://www.whdeng.cn/RAF/model1.html)
- [AffectNet](http://mohammadmahoor.com/affectnet/)
- [EmotioNet Database](http://cbcsl.ece.ohio-state.edu/dbform_emotionet.html)
- [The Radboud Faces Database (RaFD)](http://www.socsci.ru.nl:8180/RaFD2/RaFD)
- [Aff-Wild data](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/)
- A novel database of Children’s Spontaneous Facial Expressions (LIRIS-CSE) [[paper](https://arxiv.org/pdf/1812.01555.pdf)]

## FER Challenges
- [Emotion Recognition in the Wild Challenge (EmotiW) @ ICMI](https://sites.google.com/view/emotiw2018)
    + [EmotiW 2018](https://sites.google.com/view/emotiw2018)
        * [Details](https://arxiv.org/pdf/1808.07773.pdf)
        * Multi-Feature Based Emotion Recognition for Video Clips [[paper](http://delivery.acm.org/10.1145/3270000/3264989/p630-liu.pdf?ip=118.140.125.72&id=3264989&acc=OA&key=4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35%2E65B561F191013DD0&__acm__=1545619250_535c23ee84805ca9482eaf3dc8bc1590)]
        * Video-based Emotion Recognition Using Deeply-Supervised Neural Networks [[paper](https://dl.acm.org/citation.cfm?id=3264978)]
        * Multiple Spatio-temporal Feature Learning for Video-based Emotion Recognition in the Wild [[paper](https://dl.acm.org/citation.cfm?id=3264992)]
        * An Occam’s Razor View on Learning Audiovisual Emotion
Recognition with Small Training Sets [[paper](https://arxiv.org/pdf/1808.02668.pdf)]
        * Group-Level Emotion Recognition Using Hybrid Deep Models Based on Faces, Scenes, Skeletons and Visual Attentions [[paper](https://dl.acm.org/citation.cfm?id=3264990)]
        * Cascade Attention Networks For Group Emotion Recognition with Face, Body and Image Cues [[paper](https://dl.acm.org/citation.cfm?id=3264991)]
        * Group-Level Emotion Recognition using Deep Models with A Four-stream Hybrid Network [[paper](https://dl.acm.org/citation.cfm?id=3264987)]
        * An Attention Model for group-level emotion recognition[[paper](https://arxiv.org/abs/1807.03380)]
        
    + [EmotiW 2017](https://sites.google.com/site/emotiwchallenge/)
        * [Details](https://drive.google.com/file/d/1-mVVbabm8ePTMJKwO0itdMXB3j5vEw7h/view)
        * Learning supervised scoring ensemble for emotion recognition in the wild [[paper](https://dl.acm.org/citation.cfm?id=3143009)]
        * Convolutional neural networks pretrained on large face recognition datasets for emotion classification from video [[paper](https://arxiv.org/abs/1711.04598)]
        * Temporal Multimodal Fusion for Video Emotion Classification in the Wild [[paper](https://arxiv.org/pdf/1709.07200.pdf)]
        * Emotion recognition with multimodal features and temporal models [[paper](https://dl.acm.org/citation.cfm?doid=3136755.3143016)]
        * Audio-visual emotion recognition using deep transfer learning and multiple temporal models [[paper](https://dl.acm.org/citation.cfm?doid=3136755.3143012)]

    + [EmotiW 2016](https://sites.google.com/site/emotiw2016/)
    + [EmotiW 2015](https://cs.anu.edu.au/few/emotiw2015.html)
    + [EmotiW 2014](https://cs.anu.edu.au/few/emotiw2014.html)
    + [EmotiW 2013](https://cs.anu.edu.au/few/emotiw.html)
- [Audio/Visual Emotion Challenge (AVEC) @ ACM MM](https://sites.google.com/view/avec2018)
- [Facial Expression Recognition and Analysis Challenge (FERA) @ FG](http://www.fg2017.org/index.php/challenges/)
- [One-Minute Gradual-Emotion Behavior Challenge @ IJCNN](https://www2.informatik.uni-hamburg.de/wtm/OMG-EmotionChallenge/)
- [EmotioNet Challenge](http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/index.html)
- [Real Versus Fake Expressed Emotions @ ICCV](http://openaccess.thecvf.com/ICCV2017_workshops/ICCV2017_W44.py)
- [Affect-in-the-Wild Challenge @ CVPR](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/)

## Affective Level Estimation

### Dataset
- AffectNet [[Website](http://mohammadmahoor.com/affectnet/)]
- (IEEE TRANSACTIONS ON AFFECTIVE COMPUTING) AffectNet: A Database for Facial Expression,
Valence, and Arousal Computing in the Wild [[paper](https://arxiv.org/pdf/1708.03985.pdf)]
- EMOTIC dataset [[Website](http://sunai.uoc.edu/emotic/download.html)]
- (CVPR17) Emotion Recognition in Context [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Kosti_Emotion_Recognition_in_CVPR_2017_paper.pdf)][[Supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Kosti_Emotion_Recognition_in_2017_CVPR_supplemental.pdf)]
- (CVPR17 workshop) EMOTIC: Emotions in Context Dataset 
- Aff-Wild DATASET[[Website](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/)]
- Deep Affect Prediction in-the-wild: Aff-Wild Database and Challenge,
Deep Architectures, and Beyond [[paper](https://arxiv.org/pdf/1804.10938.pdf)]
- (CVPR17 workshop) Aff-Wild: Valence and Arousal ‘in-the-wild’ Challenge [[paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/cvpr_workshop_faces_in_the_wild_(1).pdf)]
- HAPPEI dataset (happiness intensity)
- DISFA [[Website](http://www.engr.du.edu/mmahoor/DISFA.htm)](AUs)
- (IEEE TRANSACTIONS ON AFFECTIVE COMPUTING13) DISFA: A Spontaneous Facial
Action Intensity Database [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=6475933)]
- GENKI-4K [[Website](http://mplab.ucsd.edu/wordpress/?page_id=398)](smile&non-smile)
- Proposed Methods validated using GENKI-4K :
- (ICCVW17) 95.76 SmileNet: Registration-Free Smiling Face Detection In The Wild [[paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Jang_SmileNet_Registration-Free_Smiling_ICCV_2017_paper.pdf)][[project](https://sites.google.com/view/sensingfeeling/)]
- (Machine Vision and Applications17) 92.05 Smile detection in the wild with deep convolutional neural
networks [[paper](https://link-springer-com.eproxy.lib.hku.hk/content/pdf/10.1007%2Fs00138-016-0817-z.pdf)]
- (ACPR15) 94.6 Facial smile detection based on deep learning features. [[paper](http://www.nlpr.ia.ac.cn/english/irds/People/lwang/M-MCG_EN/Publications/2015/KHZ2015ACPR.pdf)]
- AFFECTIVA-MIT AM-FED [[Website](https://www.affectiva.com/facial-expression-dataset/)](smile&non-smile)
- Affectiva-MIT Facial Expression Dataset (AM-FED): Naturalistic and
Spontaneous Facial Expressions Collected In-the-Wild [[paper](https://affect.media.mit.edu/pdfs/13.McDuff-etal-AMFED.pdf)]
- Binghamton
Pittsburgh 4D Spontaneous Expression datase (BP4D) [[Website](http://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html)]
- The UNBC-McMaster Shoulder Pain Expression
Archive Database


### Challenges
- Group-level happiness intensity recognition @ EmotiW16 [[Website](https://sites.google.com/site/emotiw2016/challenge-details)]
- First Affect-in-the-Wild Challenge [[Website](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/)]

### Related Works
- (ICMI16) Happiness level prediction with sequential inputs via multiple regressions [[paper](http://delivery.acm.org.eproxy.lib.hku.hk/10.1145/3000000/2997636/p487-li.pdf?ip=147.8.204.164&id=2997636&acc=ACTIVE%20SERVICE&key=CDD1E79C27AC4E65%2EDE0A32330AE3471B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1537261553_003215203b29995f2e1f0ca6091bcef6)]
- (CVPR17 workshop) Estimation of Affective Level in the Wild With Multiple Memory Networks [[paper](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w33/papers/Li_Estimation_of_Affective_CVPR_2017_paper.pdf)]
- (ICMI16) Group Happiness Assessment Using Geometric Features
and Dataset Balancing [[paper](http://vintage.winklerbros.net/Publications/emotiw2016.pdf)]
- (ICMI17) Group Emotion Recognition with Individual Facial Emotion
CNNs and Global Image Based CNNs [[paper](https://pengxj.github.io/files/icmi17-paper.pdf)]
- Feature Extraction via Recurrent Random Deep
Ensembles and its Application in Gruop-level
Happiness Estimation [[paper](https://arxiv.org/pdf/1707.09871.pdf)]
- (ICPR18) Deep Spatiotemporal Representation of the Face
for Automatic Pain Intensity Estimation [[paper](https://arxiv.org/pdf/1806.06793.pdf)]
- Learning Pain from Action Unit Combinations:
A Weakly Supervised Approach via Multiple
Instance Learning [[paper](https://arxiv.org/pdf/1712.01496.pdf)]
- (CVPR17) Personalized Automatic Estimation of Self-reported Pain Intensity
from Facial Expressions [[paper](https://arxiv.org/pdf/1706.07154.pdf)]
- (ICIP17) REGULARIZING FACE VERIFICATION NETS FOR PAIN INTENSITY REGRESSION [[paper](https://arxiv.org/pdf/1702.06925.pdf)]
- (CVPR17 Workshop) Recurrent Convolutional Neural Network Regression for Continuous Pain Intensity Estimation in Video [[paper](https://arxiv.org/pdf/1605.00894.pdf)]
- Multi-Instance Dynamic Ordinal Random Fields for
Weakly-supervised Facial Behavior Analysis [[paper](https://arxiv.org/pdf/1803.00907.pdf)]
- (FG17) Generic to Specific Recognition Models for Group Membership
Analysis in Videos [[paper](https://www.repository.cam.ac.uk/bitstream/handle/1810/274276/MouEtAl_FG17_Accepted.pdf?sequence=3)]
- (Neurocomputing18)Facial expression intensity estimation using Siamese and triplet
networks [[paper](https://ac-els-cdn-com.eproxy.lib.hku.hk/S0925231218307926/1-s2.0-S0925231218307926-main.pdf?_tid=1d157b21-6d6d-4476-8e1d-57ca2fcf8252&acdnat=1533201613_e02292ed92af320bd26447ad1d349612)]
- (Neurocomputing16) A new descriptor of gradients Self-Similarity for smile detection
in unconstrained scenarios [[paper](https://ac-els-cdn-com.eproxy.lib.hku.hk/S0925231215014812/1-s2.0-S0925231215014812-main.pdf?_tid=ecccca49-644a-46f8-a5b4-6ed4c5fced2b&acdnat=1533202351_822998183c1e602b7006e0d7f7f124f1)]
- (IEEE trans on Affective Computing11) Continuous Prediction of Spontaneous Affect from Multiple Cues and Modalities in Valence-Arousal Space [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=5740839)]


### Happiness (smile intensity estimation)
- (IEEE Transactions on Pattern Analysis and Machine Intelligence 09) Toward Practical Smile Detection [[paper](https://ieeexplore.ieee.org/document/4785473)]
- (Multimedia Tools and Applications18) Smile intensity recognition in real time videos: fuzzy system approach[[paper](https://link.springer.com/article/10.1007/s11042-018-6890-8)]
- (ACM Transactions on Intelligent Systems and Technology18) The Effect of Pets on Happiness: A Large-Scale Multi-Factor
Analysis Using Social Multimedia [[paper](http://delivery.acm.org.eproxy.lib.hku.hk/10.1145/3210000/3200751/a60-peng.pdf?ip=147.8.204.164&id=3200751&acc=ACTIVE%20SERVICE&key=CDD1E79C27AC4E65%2EDE0A32330AE3471B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1533733444_6049e63f292ec50cf0fd27cb12a9572f#URLTOKEN#)]
- (ICMSS '17) Happy Index: Analysis Based on Automatic Recognition of
Emotion Flow [[paper](http://delivery.acm.org.eproxy.lib.hku.hk/10.1145/3040000/3034961/p157-Qian.pdf?ip=147.8.204.164&id=3034961&acc=ACTIVE%20SERVICE&key=CDD1E79C27AC4E65%2EDE0A32330AE3471B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1533733721_35e65c9ce21191d0062befc7e0aa5644#URLTOKEN#)]
- (ICPR 2012)Group Expression Intensity Estimation in Videos via Gaussian Processes [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=6460925)]
- A Novel Approach to Detect Smile [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=6406670)]
- (ICCV17 Workshop) SmileNet: Registration-Free Smiling Face Detection In The Wild [[paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w23/Jang_SmileNet_Registration-Free_Smiling_ICCV_2017_paper.pdf)][[project](https://sites.google.com/view/sensingfeeling/)]
- (ACII17) Smiling from Adolescence to Old Age: A Large Observational Study [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8273585)]
- Deep Convolutional Neural Networks for Smile
Recognition [[paper](https://arxiv.org/pdf/1508.06535.pdf)]
- Embedded Implementation of a Deep Learning
Smile Detector [[paper](https://arxiv.org/pdf/1807.10570.pdf)]
- Smile detection in the wild based on transfer learning [[paper](https://arxiv.org/pdf/1802.02185.pdf)]
- Appearance-based smile intensity estimation by cascaded support vector machines [[paper](https://www.researchgate.net/publication/220745420_Appearance-Based_Smile_Intensity_Estimation_by_Cascaded_Support_Vector_Machines/download)]
- Smile detection in the wild with deep convolutional neural networks [[paper](https://link.springer.com/article/10.1007/s00138-016-0817-z)]
- Fast and Robust Smile Intensity Estimation by Cascaded
Support Vector Machines [[paper](http://www.ijcte.org/papers/640-W00031.pdf)]

### Painful Expression Intensity Estimation
- (ICMI17) Cumulative Attributes for Pain Intensity Estimation [[paper](http://delivery.acm.org.eproxy.lib.hku.hk/10.1145/3140000/3136789/icmi17-sl-1970.pdf?ip=147.8.31.43&id=3136789&acc=ACTIVE%20SERVICE&key=CDD1E79C27AC4E65%2EDE0A32330AE3471B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1546047885_fc745f38ec1bb057ab8b2a0fc1062812)]
- (CVPRW16)Recurrent Convolutional Neural Network Regression for Continuous Pain
Intensity Estimation in Video[[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7789681)][:dizzy::dizzy::dizzy:]
- (CVPRW15)Pain Recognition using Spatiotemporal Oriented Energy of Facial Muscles[[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7301340)]
- (CVPR17) Personalized Automatic Estimation of Self-reported Pain Intensity
from Facial Expressions [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8015020)][:dizzy::dizzy:]
- (FG15)Weakly Supervised Pain Localization using Multiple Instance Learning [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=6553762)]
- (ICIP17) REGULARIZING FACE VERIFICATION NETS FOR PAIN INTENSITY REGRESSION [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8296449)][[code](https://github.com/happynear/PainRegression)]
- (ICME13) PAIN DETECTION THROUGH SHAPE AND APPEARANCE FEATURES [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=6607608)]
- (ICPR14) Pain Intensity Evaluation Through Facial Action
Units [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=6977516)]
- (ieee trans on affective computing) Automatic Pain Assessment with Facial
Activity Descriptors [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7423704)]


### Facial Action Unit Estimation
- (TIP18) Facial Action Unit Recognition and Intensity Estimation Enhanced Through Label Dependencies [[paper](https://www.researchgate.net/publication/328548884_Facial_Action_Unit_Recognition_and_Intensity_Estimation_Enhanced_Through_Label_Dependencies)]
- (BMVC18) Identity-based Adversarial Training of Deep
CNNs for Facial Action Unit Recognition [[paper](http://bmvc2018.org/contents/papers/0741.pdf)]
- (ICCV17) DeepCoder: Semi-parametric Variational Autoencoders
for Automatic Facial Action Coding [[paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/tran_deepcoder_semi-parametric_variational_iccv_2017_paper.pdf)]
- (FG2017) EAC-Net: A Region-based Deep Enhancing and Cropping Approach for
Facial Action Unit Detection [[paper](https://arxiv.org/pdf/1702.02925.pdf)]
- (Journal of Visual Communication and Image Representation 17) A joint dictionary learning and regression model for intensity estimation
of facial AUs [[paper](https://ac-els-cdn-com.eproxy.lib.hku.hk/S1047320317301025/1-s2.0-S1047320317301025-main.pdf?_tid=737aee87-67c5-4beb-91bd-41f50c1c9d70&acdnat=1546048938_0bd273b5c0735d2eb21dd54502420008)]
- (TIP16) Joint Patch and Multi-label Learning for Facial
Action Unit and Holistic Expression Recognition [[paper](http://www.humansensing.cs.cmu.edu/sites/default/files/07471506.pdf)]
- (ECCV18) Deep Structure Inference Network for Facial
Action Unit Recognition [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ciprian_Corneanu_Deep_Structure_Inference_ECCV_2018_paper.pdf)]
- (FG15) Deep Learning based FACS Action Unit Occurrence and Intensity
Estimation [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7284873)]
- (TPAMI15) Discriminant Functional Learning of Color
Features for the Recognition of Facial Action
Units and their Intensities [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8454901)]
- (FG18) Edge Convolutional Network for Facial Action Intensity Estimation [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8373827)]
- (FG17) Support Vector Regression of Sparse Dictionary-Based Features for View-Independent Action Unit Intensity Estimation [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7961832)]
- (FG17) Pose-independent Facial Action Unit Intensity Regression Based on
Multi-task Deep Transfer Learning [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7961835)]
- Region-based facial representation for real-time Action Units intensity detection across datasets [[paper](https://link-springer-com.eproxy.lib.hku.hk/content/pdf/10.1007%2Fs10044-017-0645-4.pdf)]
- (ACII17) Local-Global Ranking for Facial Expression Intensity Estimation [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=8273587)]
- (FG15) How much training data for facial action unit detection? [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7163106)]
- (FG15) Facial Action Units Intensity Estimation by the Fusion of Features with
Multi-kernel Support Vector Machine [[paper](https://ieeexplore-ieee-org.eproxy.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7284870)]
- (BMVC18) Joint Action Unit localisation and intensity
estimation through heatmap regression [[paper](https://arxiv.org/pdf/1805.03487.pdf)] [[code](https://github.com/ESanchezLozano/Action-Units-Heatmaps)] [:dizzy::dizzy::dizzy::dizzy:]
- (ECCV18) Deep Adaptive Attention for Joint Facial Action
Unit Detection and Face Alignment[[paper](https://arxiv.org/pdf/1803.05588.pdf)] 
- (CVPR 18) Weakly-supervised Deep Convolutional Neural Network Learning
for Facial Action Unit Intensity Estimation [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Weakly-Supervised_Deep_Convolutional_CVPR_2018_paper.pdf)]
- (CVPR 18) Learning Facial Action Units from Web Images with
Scalable Weakly Supervised Clustering [[paper](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0237.pdf)]
- (CVPR 18) Classifier Learning with Prior Probabilities
for Facial Action Unit Recognition [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Classifier_Learning_With_CVPR_2018_paper.pdf)]
- (CVPR 18) Bilateral Ordinal Relevance Multi-instance Regression
for Facial Action Unit Intensity Estimation [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Bilateral_Ordinal_Relevance_CVPR_2018_paper.pdf)]
- (CVPR 18) Weakly Supervised Facial Action Unit Recognition through Adversarial Training [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Peng_Weakly_Supervised_Facial_CVPR_2018_paper.pdf)]
- (CVPR 2017) Deep Structured Learning for Facial Action Unit Intensity Estimation [[paper](https://arxiv.org/pdf/1704.04481.pdf)]
- (CVPR 17) Action Unit Detection with Region Adaptation, Multi-labeling Learning and
Optimal Temporal Fusing [[paper](https://arxiv.org/pdf/1704.03067.pdf)]
- (ICCV 17) Deep Facial Action Unit Recognition from Partially Labeled Data [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Wu_Deep_Facial_Action_ICCV_2017_paper.pdf)]
- (CVPR 16) Deep Region and Multi-label Learning for Facial Action Unit Detection [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Zhao_Deep_Region_and_CVPR_2016_paper.pdf)] [[code](https://github.com/zkl20061823/DRML)] [[code2](https://github.com/AlexHex7/DRML_pytorch)]
- (CVPR 16) Constrained Joint Cascade Regression Framework for Simultaneous Facial
Action Unit Recognition and Facial Landmark Detection [[paper](http://openaccess.thecvf.com/content_cvpr_2016/papers/Wu_Constrained_Joint_Cascade_CVPR_2016_paper.pdf)]
- (IEEE
Transactions on Affective Computing17) Copula Ordinal Regression Framework
for Joint Estimation of Facial Action Unit Intensity[[paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/07983431.pdf)]
- (CVPR16) Copula Ordinal Regression
for Joint Estimation of Facial Action Unit Intensity [[paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/copula_ordinal_regression__cvpr2016_final.pdf)] [[code](https://github.com/RWalecki/copula_ordinal_regression)]
- (CVPR 15) Latent Trees for Estimating Intensity of Facial Action Units [[paper](https://ibug.doc.ic.ac.uk/media/uploads/documents/kaltwang2015latent.pdf)][[code](https://github.com/kaltwang/latenttrees)]
- (ICCV 15) Learning to transfer: transferring latent task structures and its application to
person-specific facial action unit detection [[paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Almaev_Learning_to_Transfer_ICCV_2015_paper.pdf)]
-  (ICCV 15) Multi-conditional Latent Variable Model for Joint Facial Action Unit Detection [[paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Eleftheriadis_Multi-Conditional_Latent_Variable_ICCV_2015_paper.pdf)]
-  (ICCV 15) Confidence Preserving Machine for Facial Action Unit Detection [[paper](http://openaccess.thecvf.com/content_iccv_2015/papers/Zeng_Confidence_Preserving_Machine_ICCV_2015_paper.pdf)]
- Joint Patch and Multi-label Learning for Facial Action Unit Detection [[paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Zhao_Joint_Patch_and_2015_CVPR_paper.pdf)]
- (Pattern Recognition Letters14) Estimating smile intensity: A better way [[paper](https://ac-els-cdn-com.eproxy.lib.hku.hk/S0167865514003080/1-s2.0-S0167865514003080-main.pdf?_tid=466ea9bb-1959-4710-ab74-07ed82680a91&acdnat=1534058715_90dafa3071b2a1097c2252da015b3c78)] [:dizzy::dizzy::dizzy::dizzy:]
- (nccv15) Deep Learning based FACS Action Unit Occurrence and Intensity Estimation [[paper](http://www.nccv2015.nl/papers/nccv2015_p11.pdf)]
- Conditional Adversarial Synthesis of 3D Facial Action Units [[paper](https://arxiv.org/pdf/1802.07421.pdf)] [:dizzy::dizzy::dizzy:]
- (NIPS 16)Incremental Boosting Convolutional Neural Network
for Facial Action Unit Recognition [[paper](https://arxiv.org/pdf/1707.05395.pdf)]
- (Image and Vision Computing 12) Regression-based intensity estimation of facial action units [[paper](https://ac-els-cdn-com.eproxy.lib.hku.hk/S0262885611001326/1-s2.0-S0262885611001326-main.pdf?_tid=6974cc5b-a5c5-49f8-80a9-48b64d737dd0&acdnat=1545987498_086c43fef2e7b676fb897f288013d1b4)]
- (IEEE Transactions on Cybernetics 16) Intensity Estimation of Spontaneous Facial Action Units Based on Their Sparsity Properties [[paper](https://ieeexplore.ieee.org/document/7081360)]
- (TPAMI 15) Context-Sensitive Dynamic Ordinal Regression
for Intensity Estimation of Facial Action Units [[paper](https://spiral.imperial.ac.uk/bitstream/10044/1/23471/2/tpamicscorffinal_rudovic.pdf)]
- (FG 17) AUMPNet: Simultaneous Action Units Detection and Intensity Estimation on Multipose Facial Images Using a Single Convolutional Neural Network [[paper](https://www.researchgate.net/publication/315952013_AUMPNet_Simultaneous_Action_Units_Detection_and_Intensity_Estimation_on_Multipose_Facial_Images_Using_a_Single_Convolutional_Neural_Network)]
- Projects  [[Computer Expression Recognition Toolbox](http://mplab.ucsd.edu/~marni/Projects/CERT.htm)]  [[TAUD 2011](https://ibug.doc.ic.ac.uk/resources/temporal-based-action-unit-detection/)] [[LAUD 2010](https://ibug.doc.ic.ac.uk/resources/laud-programme-20102011/)] [[Openface](https://github.com/TadasBaltrusaitis/OpenFace)] [[Openface-Paper](OpenFace: an open source facial behavior analysis toolkit)]

## Attention
- (ACM MM17) Fine-Grained Recognition via Attribute-Guided Attentive
Feature Aggregation [[paper](http://delivery.acm.org.eproxy.lib.hku.hk/10.1145/3130000/3123358/p1032-yan.pdf?ip=147.8.31.43&id=3123358&acc=ACTIVE%20SERVICE&key=CDD1E79C27AC4E65%2EDE0A32330AE3471B%2E4D4702B0C3E38B35%2E4D4702B0C3E38B35&__acm__=1540166942_7e6707193038d68bed7b735df5ab6d3b)]
- (ACM MM18) visual spatial attention network for relationship detection [[paper](https://dl.acm.org/citation.cfm?id=3240611)]
- (ACM MM18) Attribute-Aware Attention Model for Fine-grained Representation Learning [[paper](https://dl.acm.org/citation.cfm?id=3240550)]
- (ACM MM18) Attention-based Multi-Patch Aggregation for Image Aesthetic Assessment [[paper](https://dl.acm.org/citation.cfm?id=3240554)]
- (ACM MM18) Attention-based Pyramid Aggregation Network
for Visual Place Recognition [[paper](https://arxiv.org/pdf/1808.00288.pdf)]
- (CVPR18) Attention-Aware Compositional Network for Person Re-identification [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Attention-Aware_Compositional_Network_CVPR_2018_paper.pdf)]
- (CVPR18) PiCANet: Learning Pixel-wise Contextual Attention for Saliency Detection [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_PiCANet_Learning_Pixel-Wise_CVPR_2018_paper.pdf)]
- (CVPR18) Emotional Attention: A Study of Image Sentiment and Visual Attention [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Fan_Emotional_Attention_A_CVPR_2018_paper.pdf)]
- (CVPR16) Learning Deep Features for Discriminative Localization [[paper](https://arxiv.org/pdf/1512.04150.pdf)][[code](https://github.com/metalbubble/CAM)]
- (CVPR18) Non-local Neural Networks [[paper](https://arxiv.org/pdf/1711.07971.pdf)]
- (ECCV18) Interaction-aware Spatio-temporal Pyramid
Attention Networks for Action Classification [[paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Yang_Du_Interaction-aware_Spatio-temporal_Pyramid_ECCV_2018_paper.pdf)]
- (ECCV18) CBAM: Convolutional Block Attention Module [[paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)]
- (CVPR18) Attentive Fashion Grammar Network for
Fashion Landmark Detection and Clothing Category Classification[[paper](http://web.cs.ucla.edu/~yuanluxu/publications/fashion_grammar_cvpr18.pdf)]
- (CVPR18) Attention-Aware Compositional Network for Person Re-identification [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Attention-Aware_Compositional_Network_CVPR_2018_paper.pdf)]
- (ICCV17) A Coarse-Fine Network for Keypoint Localization [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_A_Coarse-Fine_Network_ICCV_2017_paper.pdf)]
- Video-based Person Re-identification via 3D Convolutional Networks
and Non-local Attention [[paper](https://arxiv.org/pdf/1807.05073.pdf)]
- (AAAI19) Dual Attention Network for Scene Segmentation [[paper](https://arxiv.org/pdf/1809.02983.pdf)]
- (ECCV18) Small-scale Pedestrian Detection Based on
Topological Line Localization and Temporal
Feature Aggregation [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tao_Song_Small-scale_Pedestrian_Detection_ECCV_2018_paper.pdf)]
- (ECCV2018) CornerNet: Detecting Objects as
Paired Keypoints [[paper](https://arxiv.org/pdf/1808.01244.pdf)][[code](https://github.com/umich-vl/CornerNet)]
- (ECCV2018) DeepPhys: Video-Based Physiological
Measurement Using Convolutional
Attention Networks [[paper](https://eccv2018.org/openaccess/content_ECCV_2018/papers/Weixuan_Chen_DeepPhys_Video-Based_Physiological_ECCV_2018_paper.pdf)]
- (ECCV18) Video Object Segmentation with Joint Re-identification and
Attention-Aware Mask Propagation [[paper](https://davischallenge.org/challenge2018/papers/DAVIS-Semisupervised-Challenge-2nd-Team.pdf)]
- (ECCV18) Mancs: A Multi-task Attentional Network with
Curriculum Sampling for Person
Re-identification [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Cheng_Wang_Mancs_A_Multi-task_ECCV_2018_paper.pdf)]
- (ECCV18) Deep Imbalanced Attribute Classification using
Visual Attention Aggregation [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nikolaos_Sarafianos_Deep_Imbalanced_Attribute_ECCV_2018_paper.pdf)]
- (ECCV18) Deep Adversarial Attention Alignment for
Unsupervised Domain Adaptation:
the Benefit of Target Expectation Maximization [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Guoliang_Kang_Deep_Adversarial_Attention_ECCV_2018_paper.pdf)]
- (ECCV18) Deep Adaptive Attention for Joint Facial Action
Unit Detection and Face Alignment [[paper](https://arxiv.org/pdf/1803.05588.pdf)]



## Common Domain Adaptation 

### [[transfer Xlearn Lib code](https://github.com/thuml/Xlearn)]
- (ICML 2015) Learning Transferable Features with Deep Adaptation Networks [[paper](https://arxiv.org/pdf/1502.02791.pdf)][:dizzy::dizzy:]
- (NIPS 2016) Unsupervised Domain Adaptation with Residual Transfer Networks [[paper](https://arxiv.org/pdf/1602.04433.pdf)][:dizzy::dizzy:]
- (ICML 2017) Deep Transfer Learning with Joint Adaptation Networks [[paper](https://arxiv.org/abs/1605.06636)][:dizzy::dizzy:]

### Neurocomputing18
- Deep Visual Domain Adaptation: A Survey [[paper](https://arxiv.org/pdf/1802.03601.pdf)][:dizzy::dizzy::dizzy:]

### CVPR18
- Residual Parameter Transfer for Deep Domain Adaptation [[paper](https://arxiv.org/pdf/1711.07714.pdf)][:dizzy::dizzy:]
- Deep Cocktail Network:
Multi-source Unsupervised Domain Adaptation with Category Shift [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_Deep_Cocktail_Network_CVPR_2018_paper.pdf)][:dizzy::dizzy:]
- Detach and Adapt: Learning Cross-Domain Disentangled Deep Representation [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Detach_and_Adapt_CVPR_2018_paper.pdf)][:dizzy::dizzy::dizzy:]
- Maximum Classifier Discrepancy for Unsupervised Domain Adaptation [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf)],[[code](https://github.com/mil-tokyo/MCD_DA)][:dizzy::dizzy:]
- Adversarial Feature Augmentation for Unsupervised Domain Adaptation [[paper](https://arxiv.org/pdf/1711.08561.pdf)],[[code](https://github.com/ricvolpi/adversarial-feature-augmentation)][:dizzy::dizzy:]
- Duplex Generative Adversarial Network for Unsupervised Domain Adaptation [[paper](http://vipl.ict.ac.cn/uploadfile/upload/2018041610083083.pdf)],[[code](http://vipl.ict.ac.cn/view_database.php?id=6)][:dizzy::dizzy:]
- Generate To Adapt: Aligning Domains using Generative Adversarial Networks [[paper](https://arxiv.org/pdf/1704.01705.pdf)],[[code](https://github.com/yogeshbalaji/Generate_To_Adapt)][:dizzy::dizzy::dizzy:]
- Feature Generating Networks for Zero-Shot Learning [[paper](https://arxiv.org/pdf/1712.00981.pdf)][:dizzy::dizzy::dizzy:]

### AAAI18
-  Wasserstein Distance Guided Representation Learning
for Domain Adaptation [[paper](https://arxiv.org/pdf/1707.01217.pdf)],[[code](https://github.com/RockySJ/WDGRL)][:dizzy::dizzy:]
- Deep Asymmetric Transfer Network for Unbalanced Domain Adaptation [[paper](http://media.cs.tsinghua.edu.cn/~multimedia/cuipeng/papers/DATN.pdf)][:dizzy::dizzy:]

### ICLR18
-  A DIRT-T APPROACH TO UNSUPERVISED DOMAIN
ADAPTATION [[paper](https://arxiv.org/pdf/1802.08735.pdf)],[[code](https://github.com/RuiShu/dirt-t)][:dizzy:]

### CVPR17
- Adversarial Discriminative Domain Adaptation [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Tzeng_Adversarial_Discriminative_Domain_CVPR_2017_paper.pdf)][:dizzy::dizzy:]

### ICCV17
- Associative Domain Adaptation [[paper](https://arxiv.org/pdf/1708.00938.pdf)],[[Supplementary](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Haeusser_Associative_Domain_Adaptation_ICCV_2017_supplemental.pdf)],[[code](https://github.com/haeusser/learning_by_association)][:dizzy::dizzy::dizzy:]


## Other Cross-Domain Tasks

### CVPR18
- Deep Cost-Sensitive and Order-Preserving Feature Learning for Cross-Population Age Estimation [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Deep_Cost-Sensitive_and_CVPR_2018_paper.pdf)][:dizzy::dizzy:]
- Image-Image Domain Adaptation with Preserved Self-Similarity and Domain-Dissimilarity for Person Re-identification [[paper](https://arxiv.org/pdf/1711.07027v3.pdf)][:dizzy::dizzy::dizzy:]
- Unsupervised Cross-dataset Person Re-identification by Transfer Learning of Spatial-Temporal Patterns [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Lv_Unsupervised_Cross-Dataset_Person_CVPR_2018_paper.pdf)], [[code](https://github.com/ahangchen/TFusion)][:dizzy:]
- Conditional Generative Adversarial Network for Structured Domain Adaptation [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Hong_Conditional_Generative_Adversarial_CVPR_2018_paper.pdf)][:dizzy::dizzy::dizzy:]

### IJCAI18
- Unsupervised Cross-Modality Domain Adaptation of ConvNets for Biomedical Image Segmentations with Adversarial Loss [[paper](https://arxiv.org/pdf/1804.10916.pdf)][:dizzy::dizzy::dizzy:]

### ICCV17
-  Unsupervised domain adaptation for face recognition
in unlabeled videos [[paper](https://arxiv.org/pdf/1708.02191.pdf)][:dizzy::dizzy:]

### Others
-  Multi-task Mid-level Feature Alignment Network for Unsupervised Cross-Dataset Person Re-Identification [[paper](https://arxiv.org/pdf/1807.01440v1.pdf)][:dizzy::dizzy:]


## Image to Video Adaptation 
- (IEEE TRANSACTIONS ON CYBERNETICS 17) Semi-supervised image-to-video adaptation for video action recognition [[paper](https://ieeexplore-ieee-org.eproxy1.lib.hku.hk/stamp/stamp.jsp?tp=&arnumber=7433457)][:dizzy::dizzy:]
- (CVPR 2012) Exploiting Web Images for Event Recognition in Consumer Videos: A Multiple Source Domain Adaptation Approach [[paper](http://www.ee.columbia.edu/ln/dvmm/pubs/files/CVPR_Event.pdf)][:dizzy:]
- (IJCAI18) Exploiting Images for Video Recognition with Hierarchical Generative Adversarial Networks [[paper](https://arxiv.org/pdf/1805.04384.pdf)][:dizzy::dizzy::dizzy:]

