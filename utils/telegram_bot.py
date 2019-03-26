import telepot

chat_id = -1001481984580
token = '819065046:AAFee77GqSpq8XBzmEnAMybLqOHuy6PJ_bg'
HERA_BOT = telepot.Bot(token=token)

def send_message(message):
    HERA_BOT.sendMessage(chat_id=chat_id, text=message)




