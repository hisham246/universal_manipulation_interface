import zerorpc
client = zerorpc.Client()
client.connect("tcp://129.97.71.27:4242")

while True:
        print(client.get_joint_positions())
