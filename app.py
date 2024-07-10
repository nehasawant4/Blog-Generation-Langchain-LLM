import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers

# Function to get response from LLaMa 2 model

def getLLamaResponse(input_text, blog_style):

    # LLama2 model
    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                        model_type = 'llama',
                        config = {'max_new_tokens': 256,
                                  'temperature': 0.01})

    # Enhanced Prompt Template
    template = """
    Create a comprehensive blog post tailored for needs of a {blog_style} job profile on the topic "{input_text}". Write the blog with engaging title that captures the reader's interest and blog content under 1000 words.
    Also give references, links to credible sources, articles, or studies that support the content.
    """

    prompt = PromptTemplate(input_variables = ['blog_style', 'input_text'],
                            template=template)

    # Generate a response from LLama2 model
    response = llm.invoke(prompt.format(blog_style=blog_style, input_text=input_text))

    print(response)
    return response


st.set_page_config(page_title = 'Generate Blogs',
                   page_icon = '🤖',
                   layout = 'centered',
                   initial_sidebar_state = 'collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the blog topic")

# Creating two more columns for addtional 2 fields

blog_style = st.selectbox('Writing the blog for',
                              ('Researcher', 'Lecturer', 'Student'), index=0)

submit = st.button('Generate')

# Final Response:

if submit:
    st.write(getLLamaResponse(input_text, blog_style))