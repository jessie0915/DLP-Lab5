# DLP-Lab5
## Conditional sequence-to-sequence VAE

(The report please refer to DLP_LAB4_Report_0886035.pdf) 

#### Lab Objective
* In this lab, you need to implement a conditional seq2seq VAE for English tense conversion.
* Tense conversion (4 tenses)
  * E.g. ‘access’ to ‘accessing’, or ‘accessed’ to ‘accesses’
* Generative model
  * Gaussian noise + tense -> access, accesses, accessing, accessed



#### Lab Description
* To understand CVAE
  * Reparameterization trick
  * Log variance
  * KL lost annealing
  * Condition


#### Architecture

![Architecture](/picture/architecture.png "Architecture")

#### VAE
* VAE objective: reconstruction and generation

![VAE](/picture/VAE.png "VAE")


#### CVAE
* Reparameterization trick

![CVAE](/picture/CVAE.png "CVAE")

* Log variance
  * Output should be log variance (not variance)
* Condition
  * Simply concatenate to the hidden_0 and z
  * Embed your condition to high dimensional space (or simply use one-hot)
* KL cost annealing
  * Initially set your KL weight to 0
  * Maximum value is 1

![KL cost annealing](/picture/KL_cost_annealing.png "KL cost annealing")

#### Other details
* The encoder and decoder must be implemented by LSTM. 
* You should not adopt attention mechanism
* The loss function is nn.CrossEntropyLoss().
* The optimizer is SGD
* Adopt BLEU-4 score function in NLTK.
  * Average 10 testing socres
* Adopt Gaussian_score() to compute the generation score
  * Random sample 100 noise to generate 100 words with 4 different tenses (totally 400 words)
  * 4 words should exactly match the training data.


#### Requirements
* Modify encoder, decoder, and training functions
* Implement evaluation function, dataloader, and reparameterization trick.
* Adopt teacher-forcing and kl loss annealing in your training processing. 
* Plot the crossentropy loss, KL loss, and BLEU-4 score curve during training.
* Output examples (tense conversion & Gaussian noise with 4 tenses)

![Requirements](/picture/Requirements.png "Requirements")
