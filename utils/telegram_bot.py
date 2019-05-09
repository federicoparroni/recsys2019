import telepot
import keras

chat_id = -1001481984580
token = '819065046:AAFee77GqSpq8XBzmEnAMybLqOHuy6PJ_bg'
HERA_BOT = telepot.Bot(token=token)

def send_message(message):
    HERA_BOT.sendMessage(chat_id=chat_id, text=message)


class TelegramBotKerasCallback(keras.callbacks.Callback):

    def __init__(self, log_every_epochs=1):
        self.log_every = log_every_epochs

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
            lines.append('acc: {} - val_acc: {}'.format( round(logs['acc'],4), round(logs['val_acc'],4) ))
            lines.append('loss: {} - val_loss: {}'.format( round(logs['loss'],4), round(logs['val_loss'],4) ))
            if 'mrr' in logs:
                lines.append('mrr: {} - val_mrr: {}'.format( round(logs['mrr'],4), round(logs['val_mrr'],4) ))
            
            try:
                send_message( '\n'.join(lines) )
            except:
                pass
    
    # def on_train_end(self, logs={}):
    #     pass
    

    
