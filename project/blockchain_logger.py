from web3 import Web3

# ============================================================
# BLOCKCHAIN CONFIGURATION
# ============================================================
SEPOLIA_RPC = "https://ethereum-sepolia-rpc.publicnode.com"

CONTRACT_ADDRESS = "0x52Cb1fA1766066b010FD96f7c479Aa7a3F0F4dd4"

# Your MetaMask Sepolia private key
PRIVATE_KEY = "0e5e77874edc7c31a8297f62a7a251e58467b0fd85dc5d809b522bf9c135b772"

# ABI as native Python list — no json.loads() needed, no False/True issues
CONTRACT_ABI = [
    {
        "inputs": [
            {"internalType": "string",  "name": "_attackType",      "type": "string"},
            {"internalType": "string",  "name": "_sensorsAffected", "type": "string"},
            {"internalType": "uint256", "name": "_errorValue",      "type": "uint256"}
        ],
        "name": "logAttack",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "_id", "type": "uint256"}],
        "name": "getAttack",
        "outputs": [
            {"internalType": "uint256", "name": "timestamp",       "type": "uint256"},
            {"internalType": "string",  "name": "attackType",      "type": "string"},
            {"internalType": "string",  "name": "sensorsAffected", "type": "string"},
            {"internalType": "uint256", "name": "errorValue",      "type": "uint256"},
            {"internalType": "address", "name": "reportedBy",      "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getTotalAttacks",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "internalType": "uint256", "name": "id",              "type": "uint256"},
            {"indexed": False, "internalType": "string",  "name": "attackType",      "type": "string"},
            {"indexed": False, "internalType": "string",  "name": "sensorsAffected", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "errorValue",      "type": "uint256"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp",       "type": "uint256"}
        ],
        "name": "AttackDetected",
        "type": "event"
    }
]


class BlockchainLogger:
    def __init__(self):
        self.enabled = False
        self.w3 = None
        self.contract = None
        self.account = None

        print("=" * 70)
        print("Initializing Blockchain Logger...")

        try:
            self.w3 = Web3(Web3.HTTPProvider(SEPOLIA_RPC))

            if not self.w3.is_connected():
                print("⚠️  Could not connect to Sepolia RPC")
                print("⚠️  Blockchain logging disabled")
                return

            print(f"✓ Connected to Sepolia testnet")
            print(f"  Latest block: {self.w3.eth.block_number}")

            self.account = self.w3.eth.account.from_key(PRIVATE_KEY)
            balance = self.w3.eth.get_balance(self.account.address)
            balance_eth = self.w3.from_wei(balance, 'ether')

            print(f"✓ Wallet loaded: {self.account.address[:10]}...{self.account.address[-6:]}")
            print(f"  Balance: {balance_eth:.4f} SepoliaETH")

            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(CONTRACT_ADDRESS),
                abi=CONTRACT_ABI
            )

            total = self.contract.functions.getTotalAttacks().call()
            print(f"✓ Contract loaded: {CONTRACT_ADDRESS[:10]}...")
            print(f"  Attacks already logged on chain: {total}")

            self.enabled = True
            print("✓ Blockchain logging ACTIVE")

        except Exception as e:
            print(f"⚠️  Blockchain init error: {e}")
            print("⚠️  Blockchain logging disabled - IDS will still work")

        print("=" * 70)

    def log_attack(self, attack_type: str, sensors_affected: str, error_value: float):
        """
        Write an attack detection to Sepolia blockchain.
        Returns transaction hash string if successful, None otherwise.
        """
        if not self.enabled:
            return None

        try:
            error_int = int(error_value * 10000)
            nonce = self.w3.eth.get_transaction_count(self.account.address)

            txn = self.contract.functions.logAttack(
                attack_type,
                sensors_affected,
                error_int
            ).build_transaction({
                'chainId': 11155111,
                'gas': 200000,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
            })

            signed = self.w3.eth.account.sign_transaction(txn, private_key=PRIVATE_KEY)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            tx_hex = tx_hash.hex()

            print(f"\n🔗 BLOCKCHAIN: Attack logged!")
            print(f"   Type: {attack_type} | Sensors: {sensors_affected} | Error: {error_value:.4f}")
            print(f"   TX Hash: {tx_hex}")
            print(f"   View live: https://sepolia.etherscan.io/tx/{tx_hex}\n")

            return tx_hex

        except Exception as e:
            print(f"⚠️  Blockchain log failed: {e}")
            return None

    def get_total_logged(self) -> int:
        """Return total attacks logged on chain."""
        if not self.enabled:
            return 0
        try:
            return self.contract.functions.getTotalAttacks().call()
        except:
            return 0