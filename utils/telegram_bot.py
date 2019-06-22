import telepot
import keras

"""
To create a new private channel, you will need 2 things:
1) A token bot: you can create by BotFather inside telegram. From the bot, you have to get the 'token'
    following the on-screen commands
2) A chat id: start the bot on telegram, send any message to the bot (important!) and then go to this address:
    https://api.telegram.org/bot<yourtoken>/getUpdates
    (replace <yourtoken> with the one got at the previous step)
    Copy the id from of the chat object, ex: ... , "chat": {"id": 123456789, "first_name": ...
"""
#chat_id = -1001481984580
#token = '819065046:AAFee77GqSpq8XBzmEnAMybLqOHuy6PJ_bg'

# stores chat_id and tokens
accounts = {
    'default': (-1001481984580, '819065046:AAFee77GqSpq8XBzmEnAMybLqOHuy6PJ_bg'),
    'parro': (125016709, '716431813:AAHaKh7gsBrMoexVs1Lm7gcfHct-3Y3WT4U'),
    'gabbo': (361500321, '800854524:AAGUxIYNxcVHKyjbiQk_SbU-jWj1-3lSpEA'),
    'edo': (286935646, '675236794:AAEpSgQ44Ncs1a8nh_uvc8AXaWvspI6pz1U'),
    'teo':(295586895,'868034927:AAHdzL68dDMwO-PiaP2reI3fyfTnQXZlsVo'),
    'ale':(553968847, '890873700:AAFp-JUTR1orAxgaXtUz9rmbOludxulGXfI')
    # <insert your chat_id and token here>
}

# caches created bots per account
bots = { account:None for account in accounts.keys() }

def get_bot(account):
    """ Get or create a new bot and cache it in the dictionary.
        Return bot and chat_id
    """
    if account not in accounts:
        print('Invalid telegram bot account!')
        return None, None

    chat_id, token = accounts[account]
    if bots[account] is None:
        bots[account] = telepot.Bot(token)

    return bots[account], chat_id

def send_message(message, account='default'):
    bot, chat_id = get_bot(account)
    bot.sendMessage(chat_id=chat_id, text=message)


class TelegramBotKerasCallback(keras.callbacks.Callback):

    def __init__(self, log_every_epochs=1, account='default'):
        self.log_every = log_every_epochs
        self.account = account

    # def on_train_begin(self, logs={}):
    #     pass

    # def on_epoch_begin(self, epoch, logs={}):
    #     pass

    # def on_batch_begin(self, batch, logs={}):
    #     pass
    # def on_batch_end(self, batch, logs={}):
    #     pass

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.log_every == 0:
            lines = []
            lines.append('Epoch: {}'.format(epoch))
            if 'val_acc' in logs:
                lines.append('acc: {} - val_acc: {}'.format( round(logs['acc'],4), round(logs['val_acc'],4) ))
            else:
                lines.append('acc: {}'.format( round(logs['acc'],4) ))

            if 'val_loss' in logs:
                lines.append('loss: {} - val_loss: {}'.format( round(logs['loss'],4), round(logs['val_loss'],4) ))
            else:
                lines.append('loss: {}'.format( round(logs['loss'],4) ))

            optional_line = ''
            if 'mrr' in logs:
                optional_line = 'mrr: {}'.format( round(logs['mrr'],4) )
                if 'val_mrr' in logs:
                    optional_line += ' - val_mrr: {}'.format( round(logs['val_mrr'],4) )
            if optional_line != '':
                lines.append(optional_line)

            try:
                send_message( '\n'.join(lines), account=self.account)
            except:
                pass

    # def on_train_end(self, logs={}):
    #     pass
