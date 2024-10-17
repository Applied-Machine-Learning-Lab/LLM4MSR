# LLM4MSR: An Effective Efficient Interpretable LLM-Enhanced Paradigm for Multi-Scenario Recommendation

The implement code of LLM4MSR. Take **STAR** as backbone model and **Amazon** dataset as the illustration example.

**Preparation**

1. Install requirement packages of [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) by running:


    **'pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate'**


2. Download the [ChatGLM2-6B](https://huggingface.co/THUDM/chatglm2-6b) model from hugging face and test by running the demo on it.

3. Modify the original ource code of function of sample(beam=1) in 'lib/python3.9/site-packages/transformers/generation/utils' to output the last hidden state in order to get rid of **cuda out of memory** error.

4. Combine the file 'amazon_user_prompt_part_1.csv' and 'amazon_user_prompt_part_2.csv' into 'amazon_user_prompt.csv' in dataset folder.


**Step 1: Multi-Scenario Knowledge Reasoning**

  Run the code 'produce_llm_dict_domain.py' and 'produce_llm_dict_user.py', which take about **<2 days** for all the 24752 users on single GPU.

**Step 2: Multi-Level Knowledge Fusion**

  Run the following command:

  **'python multi_amazon.py'**

  and you can see the AUC and Logloss results on all scenarios (**Remember** to search for the learning rate, which is sensitive).
