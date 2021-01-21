#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import requests
from flask import Flask, request

import spacy
spacy.load("en_core_web_sm")
app = Flask(__name__)


@app.route('/', methods=['GET'])
def verify():

    # cuando el endpoint este registrado como webhook, debe mandar de vuelta
    # el valor de 'hub.challenge' que recibe en los argumentos de la llamada

    if request.args.get('hub.mode') == 'subscribe' \
        and request.args.get('hub.challenge'):
        if not request.args.get('hub.verify_token') \
            == os.environ['VERIFY_TOKEN']:
            return ('Verification token mismatch', 403)
        return (request.args['hub.challenge'], 200)

    return ('Hello world', 200)


@app.route('/', methods=['POST'])
def webhook():

    # endpoint para procesar los mensajes que llegan

    data = request.get_json()

    # log(data)  # logging, no necesario en produccion

    inteligente = True

    if data['object'] == 'page':

        for entry in data['entry']:
            for messaging_event in entry['messaging']:

                if messaging_event.get('message'):  # alguien envia un mensaje

                    sender_id = messaging_event['sender']['id']  # el facebook ID de la persona enviando el mensaje
                    recipient_id = messaging_event['recipient']['id']  # el facebook ID de la pagina que recibe (tu pagina)
                    message_text = messaging_event['message']['text']  # el texto del mensaje
                    test_input = messaging_event['message']['text']  # el texto del mensaje

                    if inteligente:
                        from tensorflow import keras
                        from keras.layers import Input, LSTM, Dense
                        from keras.models import Model
                        from keras.models import load_model
                        
                        entrenamiento = 'training_model.hdf5' 
                        training_model = load_model(entrenamiento)
                        encoder_inputs = training_model.input[0]
                        encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
                        encoder_states = [state_h_enc, state_c_enc]
                        encoder_model = Model(encoder_inputs, encoder_states)

                        latent_dim = 256
                        decoder_state_input_hidden = Input(shape=(latent_dim,))
                        decoder_state_input_cell = Input(shape=(latent_dim,))
                        decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
                        decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
                        decoder_states = [state_hidden, state_cell]
                        decoder_outputs = decoder_dense(decoder_outputs)
                        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

                        def decode_response(test_input):
                            #Getting the output states to pass into the decoder
                            states_value = encoder_model.predict(test_input)
                            #Generating empty target sequence of length 1
                            target_seq = np.zeros((1, 1, num_decoder_tokens))
                            #Setting the first token of target sequence with the start token
                            target_seq[0, 0, target_features_dict['<START>']] = 1.    
                            
                            #A variable to store our response word by word
                            decoded_sentence = ''
                            
                            stop_condition = False
                            while not stop_condition:
                                #Predicting output tokens with probabilities and states
                                output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
                                #Choosing the one with highest probability
                                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                                sampled_token = reverse_target_features_dict[sampled_token_index]
                                decoded_sentence += " " + sampled_token      
                                #Stop if hit max length or found the stop token
                                if (sampled_token == '<END>'): #or len(decoded_sentence) > max_decoder_seq_length):
                                    stop_condition = True
                                #Update the target sequence
                                target_seq = np.zeros((1, 1, num_decoder_tokens))
                                target_seq[0, 0, sampled_token_index] = 1.
                                #Update states
                                states_value = [hidden_state, cell_state]      
                            return decoded_sentence

                        class ChatBot:
                            negative_responses = ("no", "nope", "nah", "naw", "no gracias", "lo siento")
                            #--exit_commands = ("cerrar", "salir", "partir", "chao", "hasta luego", "parar", "detener", "terminar")
                            #Method to start the conversation
                            def start_chat(self):
                                user_response = input("Hola, soy un chatbot entrenado en diálogos de politica. ¿Te gustaría chatear conmigo?\n")
                                
                                if user_response in self.negative_responses:
                                #--print("Ok, Que tengas un gran día!")
                                    send_message(sender_id, "Ok, Que tengas un gran día!")
                                    return
                                self.chat(user_response)
                            #Method to handle the conversation
                            def chat(self, reply):
                                while not self.make_exit(reply):
                                    reply = input(self.generate_response(reply)+"\n")
                                
                            #Method to convert user input into a matrix
                            def string_to_matrix(self, user_input):
                                tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
                                user_input_matrix = np.zeros(
                                    (1, max_encoder_seq_length, num_encoder_tokens),
                                    dtype='float32')
                                for timestep, token in enumerate(tokens):
                                    if token in input_features_dict:
                                        user_input_matrix[0, timestep, input_features_dict[token]] = 1.
                                return user_input_matrix
                            
                            #Method that will create a response using seq2seq model we built
                            def generate_response(self, user_input):
                                input_matrix = self.string_to_matrix(user_input)    
                                chatbot_response = decode_response(input_matrix)    
                                #Remove <START> and <END> tokens from chatbot_response
                                chatbot_response = chatbot_response.replace("<START>",'')
                                chatbot_response = chatbot_response.replace("<END>",'')
                                #--return chatbot_response
                                send_message(sender_id, chatbot_response)   
                            
                        #--send_message(sender_id, chatbot_response)
                    else:
                        send_message(sender_id, 'Hola')

                if messaging_event.get('delivery'):  # confirmacion de delivery
                    pass

                if messaging_event.get('optin'):  # confirmacion de optin
                    pass

                if messaging_event.get('postback'):  # evento cuando usuario hace click en botones
                    pass

    return ('ok', 200)


def send_message(recipient_id, message_text):

    # log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {'access_token': os.environ['PAGE_ACCESS_TOKEN']}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({'recipient': {'id': recipient_id},
                      'message': {'text': message_text}})

    r = requests.post('https://graph.facebook.com/v2.6/me/messages',
                      params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)


def log(message):  # funcion de logging para heroku
    print(str(message))
    sys.stdout.flush()


if __name__ == '__main__':
    app.run(debug=True)
