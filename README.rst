.. raw:: html

  <div align="center">
    <h1>
      Transformer-Based Super-Resolution for Polarized Dust Emission Images
    </h1>
    This repository contains the code used for building and training the model used in 
    <a href="https://github.com/georgehalal/transformer-dust-superres/blob/main/docs/Thesis_Ch5_GHalal.pdf">Chapter 5 of my PhD thesis</a>
    for iteratively enhancing the resolution of polarized dust emission images.
    Several novel tricks are introduced in this work to achieve this with limited data, using the same model for different resolutions.
    Please read <a href="https://github.com/georgehalal/transformer-dust-superres/blob/main/docs/Thesis_Ch5_GHalal.pdf">the chapter</a>
    for details on the motivation, our approach, and the results. 
  </div>


.. contents::

==================
Model Architecture
==================
.. raw:: html

  <img src="https://github.com/georgehalal/transformer-dust-superres/blob/main/docs/images/skeleton.png" width="600px"></img>
  The model fuses information from various sources of the same patch of the sky, including Planck dust optical depth at 353 GHz (𝜏353), 
  and neutral-hydrogen-based Stokes 𝑄 and 𝑈 templates (𝑄HI and 𝑈HI), to increase the resolution of the Planck
  polarized dust emission Stokes 𝑄 and 𝑈 images at 353 GHz (𝑄LR and 𝑈LR) by a factor of 4 for various input resolutions encoded
  in the resolution embedding input.

=================
Model Predictions
=================
.. raw:: html

  <img src="https://github.com/georgehalal/transformer-dust-superres/blob/main/docs/images/predictions.png" width="600px"></img>
  Example low-resolution input (LR) and high-resolution predictions (pred) and target (HR) 353 GHz Stokes 𝑄 (top 3 rows) and 𝑈 
  (bottom 3 rows) patches of sky from the test set. The same patch of sky is shown across each column with its corresponding high angular 
  resolution denoted at the top. The colorbars are centered at zero (darkest) and brighter red (blue) corresponds to higher 
  positive (negative) values. Note the model's excellent performance across resolutions.

=============
Attention Map
=============
.. raw:: html

  <img src="https://github.com/georgehalal/transformer-dust-superres/blob/main/docs/images/attention.png" width="600px"></img> 
  This shows how important each of the 6 inputs to the 
  transformer block are to each of the 2 outputs that are passed through decoders.
  The values in each row sum up to 1 because of the softmax operation.
  A discussion on the interpretation of these values can be found in 
  <a href="https://github.com/georgehalal/transformer-dust-superres/blob/main/docs/Thesis_Ch5_GHalal.pdf">the thesis chapter</a>.

========================
References & Attribution
========================

If you make use of this code in your research, please contact halalgeorge [at] gmail [dot] com for proper citations.
