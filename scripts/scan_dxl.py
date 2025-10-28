from dynamixel_sdk import *
PORT  = "/dev/ttyACM0"   # udev 별칭 썼다면 "/dev/dynamixel"
BAUD  = 1000000            # AX-12A 등 구형은 1000000일 때 많음
PROTO = 2.0              # 구형이면 1.0
p = PortHandler(PORT); pk = PacketHandler(PROTO)
assert p.openPort() and p.setBaudRate(BAUD), "포트/보레이트 실패"
found=[]
for i in range(0,253):
    model, cr, er = pk.ping(p, i)
    if cr==COMM_SUCCESS and er==0:
        print(f"Found ID={i}, model={model}"); found.append((i,model))
if not found: print("아무것도 못 찾음: 전원/배선/BAUD/PROTO 확인")
p.closePort()