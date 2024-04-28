# Cyber_Attack_Detection_Model
This repository addresses the detection of cyber-attacks on a cyber-physical system under the framework of discrete event systems modeled with a deterministic finite automaton. 
A novel cyber-attack detection model is proposed, which captures the generated sequences of events due to the evolution of an underlying system. Based on the advantage of the automaton spatiotemporal dependencies, 
the cyber-attack detection model uses graph convolutional networks to extract spatial features of event sequences. Then, the model employs a gated recurrent unit model to re-extract spatial and temporal features from the previously extracted spatial features. 
Finally, the obtained spatial and temporal features are fed to an attention model to make the model learn the importance of different event sequences such that the model is general enough in the sense of the cyber-attacks identification accuracy and no need to identify the attack types.
Technically, the spatial and temporal features of the considered system are extracted by means of deep learning to construct the cyber-attack detection model. 
A probability threshold can be pre-defined to determine whether an event sequence is attacked by using the cyber-attack detection model. Experimental studies demonstrate the performance of the proposed methods.
