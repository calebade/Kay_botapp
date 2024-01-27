# streamlit_app_variation.py
import streamlit as st
from PIL import Image
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from streamlit_chat import message
from streamlit_option_menu import option_menu
from utils import *

# Set Streamlit page configuration
st.set_page_config(
    page_title='WellnessChat Web App',
    layout='wide'
)

# Define sidebar layout
col1, col2 = st.columns([1, 1])

# Sidebar content
with col1:
    logo = Image.open("WellnessChatLogo.png")
    st.image(logo, width=430)

with col2:
    st.subheader("Team WellnessX, Project Health Hub 2.0")

# Main content
st.header(" ")

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title='Main menu',
        options=['About', 'Chatbot', 'Fitness Tips', 'Nutrition Tips'],
        icons=['house-fill', 'chat-fill', 'heart-fill', 'apple-fill'],
        menu_icon="cast",
        default_index=0,
    )

# Handle selected menu option
if selected == "About":
    st.write(" ")
    st.header(":blue[Project Background]")
    st.write("""
             WellnessChat is a virtual assistant designed to provide information and support for maintaining a healthy lifestyle. 
             The focus is on fitness, nutrition, and overall well-being. Our goal is to empower users with knowledge and guidance 
             to make informed choices for a healthier and happier life.
             """)

    st.header(":blue[Mission Statement]")
    st.write("""
             Our mission is to create a supportive and informative space where individuals can learn about and embrace a healthy lifestyle. 
             We aim to provide personalized advice, tips, and resources to help users on their journey to wellness.
             """)

    st.write(" ")

elif selected == "Chatbot":
    # Initialize chatbot components
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Hi there! How can I assist you with your wellness journey today?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=st.secrets["OPENAI_API_KEY"])

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

    system_msg_template = SystemMessagePromptTemplate.from_template(template="""
    Welcome to WellnessChat! I'm here to provide information and support for your wellness journey. 
    Feel free to ask me anything related to fitness, nutrition, or general well-being. 
    Let's work together to achieve your health goals!
    """)

    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

    # Container for chat history
    response_container = st.container()
    # Container for text box
    textcontainer = st.container()

    with textcontainer:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                st.code(conversation_string)
                refined_query = query_refiner(conversation_string, query)
                context = find_match(refined_query)
                response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
            st.session_state.requests.append(query)
            st.session_state.responses.append(response)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

elif selected == "Fitness Tips":
    st.write("""
    Here are some fitness tips to help you stay active and maintain a healthy lifestyle:
    1. Incorporate both cardiovascular exercises (e.g., running, cycling) and strength training into your routine.
    2. Find activities you enjoy to make exercise more enjoyable and sustainable.
    3. Set realistic fitness goals and track your progress.
    4. Stay hydrated before, during, and after your workouts.
    5. Ensure proper form during exercises to prevent injuries.
    6. Include flexibility and stretching exercises to improve overall mobility.
    7. Listen to your body and take rest days as needed.
    Remember, consistency is key for long-term fitness success!
    """)

elif selected == "Nutrition Tips":
    st.write("""
    Good nutrition is essential for overall well-being. Here are some nutrition tips for a healthy lifestyle:
    1. Eat a variety of fruits and vegetables for a range of nutrients.
    2. Include whole grains, lean proteins, and healthy fats in your diet.
    3. Stay mindful of portion sizes to maintain a balanced intake.
    4. Hydrate with water throughout the day.
    5. Limit added sugars and processed foods.
    6. Plan meals ahead to make healthier choices.
    7. Listen to your body's hunger and fullness cues.
    Remember, nourishing your body with wholesome foods contributes to better health!
    """)

st.markdown(
    "`Created by` Team WellnessX | 2024 | \
    `Code:` [Github](https://github.com/YourGitHubUsername/WellnessChat)"
)
