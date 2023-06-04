import json
import matplotlib.pyplot as plt
import PIL


with open("runs_recap.json", "r") as file:
    data_model = json.load(file)



model_and_architecture = [i for i in data_model]#[data_model[i]['config']['architecture'] for i in data_model]

# for model in model_and_architecture:
#     test_l = data_model[model]['test_loss']
    
#     plt.plot([i for i in range(len(test_l))], test_l, label = model)
   
# plt.xlabel('Epochs')
# plt.ylabel('Test error')
# plt.legend(title = 'Models', loc = 'center right')
# plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex = True)
ax1.set_title('Test error')
ax2.set_title('Train error')
ax3.set_title('Validation error')
for model in model_and_architecture:
    test_l = data_model[model]['test_loss']
    train_l = data_model[model]['train_loss']
    val_l = data_model[model]['val_loss']
    
    
    ax1.plot([i for i in range(len(test_l))], test_l, label = model)
    ax2.plot([i for i in range(len(train_l))], train_l, label = model)
    ax3.plot([i for i in range(len(val_l))], val_l, label = model)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.text((2.0, 2.0), 'Running results')
plt.legend(title = 'Models', loc='upper right', bbox_to_anchor=(0.6, 3, 0.5, 0.5))
plt.savefig('Models running results.png')
plt.show()

