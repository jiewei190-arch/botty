# Oracle Cloud Always Free deployment

This deployment runs the Botty worker and dashboard continuously in one Docker
container. Systemd restarts the container after a crash or VM reboot, and the
SQLite database, cache, tracker history, and logs persist under
`/var/lib/botty` on the VM boot volume.

## VM settings

Create one Always Free compute instance in the account's **home region**:

- Image: Ubuntu 24.04 (Always Free eligible)
- Shape: `VM.Standard.A1.Flex` (Ampere Arm)
- Resources: 1 OCPU and 2 GB RAM
- Boot volume: the default 50 GB
- Networking: public IPv4 address assigned
- Ingress: TCP 22 from your own IP and TCP 80 for the dashboard

If A1 capacity is unavailable, try another availability domain or use the
Always Free `VM.Standard.E2.1.Micro` shape. The Docker image supports both Arm
and AMD64.

## Install

SSH into the VM and run:

```bash
git clone --branch codex/order-reconciliation \
  https://github.com/jiewei190-arch/botty.git
cd botty
sudo bash deploy/oracle-cloud/install.sh
sudo bash deploy/oracle-cloud/configure.sh
sudo /opt/botty/deploy/oracle-cloud/start.sh
```

In the private environment file, fill these four values:

- `ALPACA_API_KEY` (paper account)
- `ALPACA_SECRET_KEY` (paper account)
- `AUTO_WEBHOOK_URL` (Slack `#general` incoming webhook)
- `DASHBOARD_PASSWORD` (a new strong password for the public dashboard)

Do not change `TRADING_MODE=paper` or `ENABLE_LIVE_TRADING=false`.

Open `http://PUBLIC_IP` and enter the dashboard password. Check service health
and logs with:

```bash
sudo systemctl status botty
sudo journalctl -u botty -f
```

Stop the Render service before starting this service. Running two Botty workers
against the same Alpaca paper account can produce duplicate scans or orders.

## Updating

From the original checkout:

```bash
git pull --ff-only
sudo bash deploy/oracle-cloud/install.sh
sudo systemctl restart botty
```

The installer never overwrites `/etc/botty/botty.env` or `/var/lib/botty`.
