sudo sysctl -w net.ipv4.ip_forward=1
mm-delay 40 mm-link traces/TMobile-UMTS-driving-cut.up traces/TMobile-UMTS-driving-cut.down --downlink-queue=droptail --downlink-queue-args="bytes=150000" --uplink-queue=infinite
