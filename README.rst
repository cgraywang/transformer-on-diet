Transformer on a Diet
-----------------------------------

Reference: C Wang, Z Ye, A Zhang, Z Zhang, A Smola. "`Transformer on a Diet <https://arxiv.org/abs/2002.06170>`_". arXiv preprint arXiv (2020).

Installation
~~~~~~~~~~~~~~~~

.. code::

    pip install --pre --upgrade mxnet
    python setup.py install

Results
~~~~~~~~~~~~~~~~

The results and the command line to reproduce the results on PTB dataset are as follows.

.. editing URL for the following table: https://tinyurl.com/w62s5s9

[1] Full (Val PPL 109.19 Test PPL 103.72)

.. code-block:: console

   $ cd scripts/language_model/
   $ python transformer_language_model.py --model full --data ptb --emsize 320 --nhid 2000 --nlayers 3 --lr 10 --epochs 500 --batch_size 20 --bptt 70 --dropout 0.4 --dropout_h 0.25 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --alpha 0 --beta 0 --lr_update_interval 100 --lr_update_factor 1 --num_heads 16 --scaled --units 320 --use_residual --max_src_length 1000 --warmup_steps 0 --first_window_size 1 --kernel_size 3 --d_base 2

[2] Dilated (Val PPL 115.67 Test PPL 110.92)

.. code-block:: console

   $ cd scripts/language_model/
   $ python transformer_language_model.py --model dilated --data ptb --emsize 320 --nhid 2000 --nlayers 3 --lr 10 --epochs 500 --batch_size 20 --bptt 70 --dropout 0.4 --dropout_h 0.25 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --alpha 0 --beta 0 --lr_update_interval 100 --lr_update_factor 1 --num_heads 16 --scaled --units 320 --use_residual --max_src_length 1000 --warmup_steps 0 --first_window_size 1 --kernel_size 3 --d_base 2

[3] Dilated-Memory (Val PPL 115.35 Test PPL 110.98)

.. code-block:: console

   $ cd scripts/language_model/
   $ python transformer_language_model.py --model dilated_mem --data ptb --emsize 320 --nhid 2000 --nlayers 3 --lr 10 --epochs 500 --batch_size 20 --bptt 70 --dropout 0.4 --dropout_h 0.25 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --alpha 0 --beta 0 --lr_update_interval 100 --lr_update_factor 1 --num_heads 16 --scaled --units 320 --use_residual --max_src_length 1000 --warmup_steps 0 --first_window_size 1 --kernel_size 3 --d_base 2

[4] Cascade (Val PPL 109.16 Test PPL 105.27)

.. code-block:: console

   $ cd scripts/language_model/
   $ python transformer_language_model.py --model cascade --data ptb --emsize 320 --nhid 2000 --nlayers 3 --lr 10 --epochs 500 --batch_size 20 --bptt 70 --dropout 0.4 --dropout_h 0.25 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --alpha 0 --beta 0 --lr_update_interval 100 --lr_update_factor 1 --num_heads 16 --scaled --units 320 --use_residual --max_src_length 1000 --warmup_steps 0 --first_window_size 4 --window_size_multiplier 2 --kernel_size 3 --d_base 2

Note that the command to reproduce the results on wikitext-2 would be updated soon.
And the repo would be cleaned further to contain only relevant modules.

Reference Paper
~~~~~~~~~~~~~~~~

The bibtext entry of the `reference paper <https://arxiv.org/abs/2002.06170>`_ is:

.. code::

   @article{transformerdiet2020,
      title={Transformer on a Diet},
      author={Chenguang Wang and Zihao Ye and Aston Zhang and Zheng Zhang and Alexander J. Smola},
      journal={ArXiv},
      year={2020},
      volume={abs/2002.06170}
   }
