from libs.training import training,simulation,getInputs,createInputs
#training()
#simulation()
entradas,out=getInputs()
# print("len",len(entradas[0]))
print("salidas",out)
# training(inputs=entradas,outputs=out,numPatterns=len(entradas))

# if (len(entradas[0])==22050):
#     simulation(output=entradas[4])