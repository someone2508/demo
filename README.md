# 🤖 AI Chat Interface

A modern, feature-rich AI chat application built with **vanilla HTML, CSS, and JavaScript**. This production-ready application showcases advanced JavaScript concepts and provides a beautiful user interface for interacting with OpenAI's GPT models.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![JavaScript](https://img.shields.io/badge/JavaScript-ES6+-yellow.svg)
![No Dependencies](https://img.shields.io/badge/dependencies-none-green.svg)

## ✨ Features

### 🎯 Core Features
- **Multiple Conversation Threads** - Create and manage multiple chat conversations
- **Real-time Streaming Responses** - Stream AI responses in real-time for better UX
- **Syntax Highlighting** - Beautiful code highlighting in chat responses
- **Markdown Support** - Full markdown rendering with code blocks
- **Chat History** - Persistent storage of all conversations in localStorage
- **Export Conversations** - Export chats as text files

### 🚀 Advanced JavaScript Features Used
- **ES6+ Classes** - Object-oriented architecture with classes
- **Async/Await** - Modern asynchronous programming
- **Promises** - Proper promise handling and error management
- **LocalStorage API** - Client-side data persistence
- **Fetch API** - HTTP requests to OpenAI API
- **Streaming API** - ReadableStream for real-time data streaming
- **Event Delegation** - Efficient event handling
- **Template Literals** - Dynamic HTML generation
- **Destructuring** - Clean parameter handling
- **Arrow Functions** - Concise function syntax
- **Array Methods** - map, filter, reduce, find, etc.
- **Object Methods** - Object manipulation and utilities
- **Closures** - Encapsulation and private variables
- **Module Pattern** - Code organization and separation
- **Observer Pattern** - Event-driven architecture
- **Factory Pattern** - Object creation
- **Debouncing** - Performance optimization
- **Error Handling** - try/catch with proper error management
- **Regular Expressions** - Text parsing and validation
- **DOM Manipulation** - Dynamic UI updates
- **Web Storage API** - localStorage management
- **Custom Events** - Application events
- **Animation API** - CSS animations via JavaScript

### 🎨 UI/UX Features
- **Modern Dark/Light Theme** - Toggle between themes
- **Responsive Design** - Works on all screen sizes
- **Smooth Animations** - CSS animations and transitions
- **Toast Notifications** - Beautiful notification system
- **Typing Indicators** - Real-time typing feedback
- **Character Counter** - Message length tracking
- **Auto-resize Textarea** - Dynamic input field
- **Keyboard Shortcuts** - Enter to send, Shift+Enter for new line
- **Modal Dialogs** - Settings configuration
- **Loading States** - Visual feedback during operations
- **Error States** - Clear error messaging

### ⚙️ Configurable Settings
- API Key management
- Model selection (GPT-4, GPT-3.5 Turbo, etc.)
- Temperature control
- Max tokens configuration
- Streaming toggle
- Sound effects toggle
- Theme selection

## 📋 Prerequisites

- A modern web browser (Chrome, Firefox, Safari, Edge)
- An OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

## 🚀 Quick Start

### 1. Clone or Download

Download the files or clone this repository:
```bash
git clone <repository-url>
cd ai-chat-interface
```

### 2. Open in Browser

Simply open `index.html` in your web browser:
```bash
# On macOS
open index.html

# On Linux
xdg-open index.html

# On Windows
start index.html
```

Or use a local server (recommended):
```bash
# Using Python 3
python -m http.server 8000

# Using Node.js (http-server)
npx http-server

# Using PHP
php -S localhost:8000
```

Then visit `http://localhost:8000`

### 3. Configure API Key

1. Click the **Settings** button (gear icon) in the sidebar
2. Enter your OpenAI API key
3. Configure other settings as needed
4. Click **Save Settings**

### 4. Start Chatting!

- Click **New Conversation** to start
- Type your message in the input box
- Press **Enter** to send (or **Shift+Enter** for new line)
- Watch the AI respond in real-time!

## 📁 File Structure

```
ai-chat-interface/
│
├── index.html          # Main HTML structure
├── styles.css          # Complete styling with animations
├── app.js              # Full application logic (2000+ lines)
└── README.md           # This file
```

## 🔧 Configuration

### API Settings

The application supports multiple OpenAI models:
- GPT-4
- GPT-4 Turbo
- GPT-3.5 Turbo (default)

### Adjustable Parameters

- **Temperature** (0.0 - 2.0): Controls randomness
  - Lower values = more focused and deterministic
  - Higher values = more random and creative
  
- **Max Tokens** (100 - 4000): Maximum response length
  
- **Streaming**: Enable/disable real-time response streaming

## 💾 Data Storage

All data is stored locally in your browser's localStorage:
- **Conversations**: All chat messages and metadata
- **Settings**: API key and preferences
- **No server required**: Everything runs client-side

**Privacy Note**: Your API key and conversations never leave your browser except when communicating with OpenAI's API.

## 🎯 Advanced Usage

### Keyboard Shortcuts

- `Enter` - Send message
- `Shift + Enter` - New line in message
- `Esc` - Close modal (when open)

### Example Prompts

Try these example prompts to get started:
- "Explain quantum computing in simple terms"
- "Write a Python function to reverse a string"
- "What are the best practices for web accessibility?"
- "Help me debug this JavaScript code"

### Exporting Conversations

Click the **Export** button (download icon) to save your conversation as a text file.

### Managing Conversations

- **New Conversation**: Create a fresh chat thread
- **Switch Conversations**: Click on any conversation in the sidebar
- **Delete Conversation**: Click the trash icon in the chat header
- **Clear All**: Remove all conversations at once

## 🎨 Customization

### Changing Colors

Edit the CSS variables in `styles.css`:

```css
:root {
    --primary-color: #10a37f;
    --secondary-color: #6366f1;
    --background-primary: #0f0f0f;
    /* ... more variables */
}
```

### Adding Custom Models

Edit the model select options in `index.html`:

```html
<select id="modelSelect">
    <option value="gpt-4">GPT-4</option>
    <option value="your-custom-model">Your Model</option>
</select>
```

### Modifying Behavior

All application logic is well-organized in `app.js` with clear class structures:
- `Utils` - Utility functions
- `StorageManager` - localStorage operations
- `ToastManager` - Notifications
- `SettingsManager` - Settings handling
- `Message` - Message model
- `Conversation` - Conversation model
- `OpenAIClient` - API communication
- `ConversationManager` - Conversation management
- `UI` - User interface controller
- `App` - Application initialization

## 🐛 Debugging

The application includes a debug API accessible in the browser console:

```javascript
// Access debug information
window.DebugAPI.conversations     // View all conversations
window.DebugAPI.settings          // View current settings
window.DebugAPI.currentConversation()  // Get active conversation
window.DebugAPI.utils             // Utility functions
```

## 🔒 Security

- API keys are stored in localStorage (client-side only)
- All user input is properly escaped to prevent XSS
- No server-side code or database required
- HTTPS recommended for production deployment

## 🚨 Troubleshooting

### API Key Issues
**Problem**: "API key not configured"
**Solution**: Add your OpenAI API key in Settings

### Network Errors
**Problem**: "Failed to get response"
**Solutions**:
- Check your internet connection
- Verify your API key is valid
- Check OpenAI API status
- Ensure you have API credits

### Storage Issues
**Problem**: Conversations not saving
**Solutions**:
- Check browser localStorage is enabled
- Clear browser cache and try again
- Check browser storage quota

### Display Issues
**Problem**: UI looks broken
**Solutions**:
- Hard refresh the page (Ctrl+Shift+R)
- Clear browser cache
- Check browser console for errors
- Ensure JavaScript is enabled

## 📊 Browser Support

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🎓 Learning Resources

This project demonstrates:

1. **Modern JavaScript**
   - Classes and OOP
   - Async programming
   - Module patterns
   - Error handling

2. **Web APIs**
   - Fetch API
   - Storage API
   - DOM API
   - Clipboard API

3. **CSS Techniques**
   - CSS Grid and Flexbox
   - CSS Variables
   - Animations
   - Responsive design

4. **Best Practices**
   - Code organization
   - Error handling
   - User feedback
   - Accessibility

## 🤝 Contributing

Feel free to fork this project and make improvements! Some ideas:
- Add voice input/output
- Support for image uploads
- Multi-language support
- Custom themes
- Plugin system
- File attachments
- Search functionality

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [OpenAI](https://openai.com) - For the amazing GPT models
- [Highlight.js](https://highlightjs.org) - Code syntax highlighting
- [Marked.js](https://marked.js.org) - Markdown parsing

## 📬 Support

If you encounter any issues or have questions:
1. Check the Troubleshooting section
2. Review the browser console for errors
3. Ensure your API key is valid
4. Check OpenAI API documentation

## 🎉 Credits

Built with ❤️ using vanilla HTML, CSS, and JavaScript.

---

**Happy Chatting! 🚀**
