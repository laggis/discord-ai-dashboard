# AI Discord Bot with Web Dashboard

A powerful Discord bot with AI capabilities, user management, and a modern web dashboard. Features include chat moderation, user management, and real-time configuration.

## 🌟 Features

### 🤖 AI Integration
- Natural language processing for chat interactions
- Smart response generation
- Learning system for improved responses

### 👥 User Management
- Secure user authentication
- Role-based access control (Admin/User)
- Password management with secure hashing
- Session management with "Remember Me" functionality

### 🛡️ Content Moderation
- Real-time message moderation using Hugging Face AI API
- Context-aware moderation that recognizes technical discussions
- Role-based moderation bypass using Discord role IDs
- Multiple word list categories as fallback:
  - Profanity
  - Hate speech
  - Mild profanity
  - Custom words
- Smart severity detection:
  - Reduces false positives for technical discussions
  - Recognizes gaming and FiveM-related terminology
  - Adjusts severity based on context
- Progressive punishment system
- Moderation action logging

#### Setting Up Moderation
1. Get a Hugging Face API key from https://huggingface.co/settings/tokens
2. Add it to your `.env` file:
   ```
   HUGGINGFACE_API_KEY=your_api_key_here
   ```
3. Configure bypass roles in `config.yaml`:
   ```yaml
   moderation:
     roles:
       bypass_moderation_ids:
         - 123456789  # Replace with your Discord role IDs
   ```
4. To get a Discord role ID:
   - Enable Developer Mode in Discord
   - Right-click the role
   - Click 'Copy ID'

### 🎛️ Web Dashboard
- Modern, responsive design with Bootstrap
- Real-time statistics
- User management interface
- Moderation control panel
- Configuration management

## 📋 Requirements

```
Python 3.8+
Discord.py
Flask
PyYAML
Tesseract-OCR
```

### Installing Tesseract-OCR
1. Download the Tesseract installer from the [UB-Mannheim repository](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer (e.g., `tesseract-ocr-w64-setup-5.3.1.20230401.exe`)
3. During installation:
   - Remember the installation path (default is `C:\Program Files\Tesseract-OCR`)
   - Add Tesseract to the system PATH when prompted
4. Verify installation by opening a new command prompt and typing:
   ```bash
   tesseract --version
   ```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/laggis/discord-ai-dashboard.git
cd discord-ai-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure the bot**
- Copy `config.yaml.example` to `config.yaml`
- Add your Discord bot token
- Configure other settings as needed

4. **Configure the start batch file**
- Open `start ai.bat` in a text editor
- Change the directory path to match your installation:
  ```batch
  cd /d "C:\Path\To\Your\Bot\Directory"
  ```

5. **Start the bot and web server**
- Double-click `start ai.bat` to run both the bot and web server
- Or run them separately:
  ```bash
  python web_config.py
  python AI.py
  ```

5. **Access the dashboard**
- Open http://localhost:5006 in your browser
- Login with default credentials:
  - Username: admin
  - Password: admin
- Change the default password immediately!

## 🔧 Configuration

### Discord Bot Settings
```yaml
discord:
  token: "YOUR_BOT_TOKEN"
  prefix: "!"
```

### Moderation Settings
```yaml
moderation:
  enabled: true
  settings:
    severity_levels:
      low: 0.3    # Mild profanity
      medium: 0.6  # Regular profanity
      high: 0.8    # Hate speech
    auto_moderation:
      max_violations: 3
      timeout_duration: 300
```

## 🛡️ Security Features

- Password hashing with PBKDF2
- Session encryption
- CSRF protection
- Role-based access control
- Secure configuration management

## 📊 Dashboard Features

### User Management
- Add/remove users
- Change passwords
- Assign roles
- View user activity

### Moderation Dashboard
- Real-time statistics
- Word list management
- Configure severity levels
- View moderation logs
- Hot-reload configuration

## 🔍 Monitoring

The bot includes comprehensive logging:
- Moderation actions
- User activities
- System events
- Configuration changes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Discord.py for the excellent Discord API wrapper
- Flask for the web framework
- Bootstrap for the UI components

## 🔮 Future Plans

- [ ] Multi-language support
- [ ] Advanced AI training capabilities
- [ ] Custom plugin system
- [ ] API integration
- [ ] Enhanced analytics

## ⚠️ Important Notes

- Always change the default admin password
- Keep your Discord bot token secure
- Regularly backup your configuration
- Monitor the moderation logs

## 🆘 Support

For support:
1. Check the [Issues](https://github.com/laggis/discord-ai-dashboard/issues) page
2. Create a new issue if needed
3. Join our Discord server for community support

## 🔄 Updates

The bot supports hot-reloading of configurations:
- Word list changes apply instantly
- Severity thresholds update in real-time
- Moderation settings can be changed without restart
- User management changes take effect immediately
