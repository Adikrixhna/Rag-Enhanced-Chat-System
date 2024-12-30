css = '''
<style>
body {
    background-color: #f5f5f5;
    font-family: 'Arial', sans-serif;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.chat-message.user {
    background-color: #007bff;
    color: white;
}
.chat-message.bot {
    background-color: #e9ecef;
    color: black;
}
.chat-message .avatar {
    width: 10%;
}
.chat-message .avatar img {
    max-width: 50px;
    max-height: 50px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 90%;
    padding: 0 1rem;
}
.stTextInput, .stButton {
    margin-top: 1rem;
    width: 100%;
}
.stTextInput textarea {
    width: 100%;
    border-radius: 0.5rem;
    padding: 1rem;
    border: 1px solid #ced4da;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.stButton button {
    background-color: #007bff;
    color: white;
    border: none;
    border-radius: 0.5rem;
    padding: 0.75rem;
    cursor: pointer;
    transition: background-color 0.3s;
}
.stButton button:hover {
    background-color: #0056b3;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="./images/Graident Ai Robot.jpg" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="./images/user.avif" alt="User Avatar">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''