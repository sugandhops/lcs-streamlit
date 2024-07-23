# LCS Streamlit

This repository contains a Streamlit application that uses LangChain and Pinecone for document retrieval and natural language processing.

## Clone the Repository

To get started, clone the repository using:

```bash
git clone https://github.com/sugandhops/lcs-streamlit.git

Set Up Environment Variables

Create a .env file in the root directory of the project with the following content:

PINECONE_API_KEY=xxx
HUGGINGFACE_API_KEY=xxx

Replace xxx with your actual API keys.


Install Dependencies

Ensure you have Python and pip installed. Then, install the required Python packages:

pip install -r requirements.txt


Run the Application

streamlit run streamlit_app.py
