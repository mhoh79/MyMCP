# Skeleton MCP Server Template

A minimal, working template for creating new MCP servers. This template demonstrates the essential patterns and best practices for building MCP servers using the BaseMCPServer framework.

## 📋 Overview

This skeleton server provides:
- ✅ Two example tools (echo and reverse)
- ✅ Comprehensive inline documentation
- ✅ Proper error handling patterns
- ✅ Support for both stdio and HTTP transports
- ✅ Type-safe implementation
- ✅ Logging and validation

## 🎯 What's Included

### Tools

1. **echo** - Returns input text unchanged
   - Demonstrates basic input/output handling
   - Shows validation patterns
   - Example of simple tool implementation

2. **reverse** - Reverses input text
   - Demonstrates string manipulation
   - Shows data transformation patterns
   - Example of processing logic

## 🚀 Quick Start Guide

### Step 1: Test the Skeleton Server

Before customizing, verify that the skeleton server works:

```bash
# Test in stdio mode (default)
cd /home/runner/work/MyMCP/MyMCP
python -m src.templates.skeleton_server.server

# Test in HTTP mode
python -m src.templates.skeleton_server.server --transport http --port 8000
```

### Step 2: Copy to Custom Directory

Copy this template to create your own server:

```bash
# Copy the entire skeleton_server directory
cp -r src/templates/skeleton_server src/custom/my_server

# Navigate to your new server
cd src/custom/my_server
```

### Step 3: Customize Your Server

Edit `server.py` in your new server directory:

1. **Rename the class** (line 34):
   ```python
   class MyServer(BaseMCPServer):  # Change from SkeletonServer
   ```

2. **Update server name** (line 221):
   ```python
   def get_server_name(self) -> str:
       return "my-server"  # Use lowercase with hyphens
   ```

3. **Update server version** (line 235):
   ```python
   def get_server_version(self) -> str:
       return "1.0.0"  # Use semantic versioning
   ```

4. **Update main() description** (line 248):
   ```python
   parser = BaseMCPServer.create_argument_parser(
       description="My Custom MCP Server - Does amazing things"
   )
   ```

### Step 4: Add Your Own Tools

Replace the example tools with your own:

1. **Define your tool** in `register_tools()`:
   ```python
   my_tool = Tool(
       name="my_tool_name",
       description="What this tool does",
       inputSchema={
           "type": "object",
           "properties": {
               "param1": {
                   "type": "string",
                   "description": "Description of parameter"
               }
           },
           "required": ["param1"]
       }
   )
   ```

2. **Create a handler function**:
   ```python
   async def handle_my_tool(arguments: dict) -> CallToolResult:
       try:
           # Extract parameters
           param1 = arguments.get("param1", "")
           
           # Validate input
           if not param1:
               return CallToolResult(
                   content=[TextContent(
                       type="text",
                       text="Error: param1 is required"
                   )],
                   isError=True
               )
           
           # Process the request
           result = do_something_with(param1)
           
           # Return success
           return CallToolResult(
               content=[TextContent(
                   type="text",
                   text=f"Result: {result}"
               )],
               isError=False
           )
       except Exception as e:
           self.logger.error(f"Error in my_tool: {e}", exc_info=True)
           return CallToolResult(
               content=[TextContent(
                   type="text",
                   text=f"Error: {str(e)}"
               )],
               isError=True
           )
   ```

3. **Register the tool**:
   ```python
   self.tool_registry.register_tool(my_tool, handle_my_tool)
   ```

### Step 5: Test Your Server

```bash
# Run your custom server
python -m src.custom.my_server.server

# Test in HTTP mode
python -m src.custom.my_server.server --transport http --port 8001
```

### Step 6: Configure for Claude Desktop

Add your server to Claude Desktop's MCP configuration:

```json
{
  "mcpServers": {
    "my-server": {
      "command": "python",
      "args": [
        "-m",
        "src.custom.my_server.server"
      ],
      "cwd": "/absolute/path/to/MyMCP"
    }
  }
}
```

## 📚 Key Concepts

### BaseMCPServer

The base class provides:
- Automatic transport handling (stdio/HTTP)
- Built-in endpoints (/health, /ready, /metrics)
- Logging and error handling
- Configuration management
- Tool registry
- Server state tracking

### Tool Definition

Every tool needs:
1. **Name**: Unique identifier (lowercase, underscores)
2. **Description**: Clear explanation of what it does
3. **Input Schema**: JSON Schema for parameters
   - Define types (string, number, boolean, array, object)
   - Mark required parameters
   - Add descriptions for each parameter

### Handler Function

Handler functions must:
1. Be async functions
2. Accept `arguments` dict parameter
3. Return `CallToolResult` object
4. Handle errors gracefully
5. Validate inputs
6. Log errors for debugging

### Error Handling

Always return proper error responses:
```python
return CallToolResult(
    content=[TextContent(type="text", text="Error message")],
    isError=True
)
```

## 🏗️ Project Structure

```
src/custom/my_server/
├── __init__.py          # Package initialization (import your server)
├── server.py            # Main server implementation
└── README.md            # Documentation for your server
```

## 📝 Best Practices

1. **Input Validation**: Always validate user inputs
2. **Error Messages**: Provide clear, helpful error messages
3. **Logging**: Use `self.logger` for debugging
4. **Documentation**: Comment your code and update the README
5. **Type Hints**: Use type hints for better IDE support
6. **Testing**: Test your tools before deploying

## 🔍 Common Patterns

### String Processing
```python
text = arguments.get("text", "")
result = text.upper()  # or lower(), strip(), etc.
```

### Number Processing
```python
value = arguments.get("value", 0)
if not isinstance(value, (int, float)):
    return error_result("value must be a number")
result = value * 2
```

### List Processing
```python
items = arguments.get("items", [])
if not isinstance(items, list):
    return error_result("items must be a list")
result = [process(item) for item in items]
```

### Conditional Logic
```python
option = arguments.get("option", "default")
if option == "A":
    result = do_a()
elif option == "B":
    result = do_b()
else:
    result = do_default()
```

## 🎓 Learning Resources

- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [BaseMCPServer Documentation](../../core/README.md)
- [Example Servers](../../builtin/)
  - Math Server: Mathematical calculations
  - Stats Server: Statistical analysis

## ❓ Troubleshooting

### Server won't start
- Check Python version (3.8+)
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check for syntax errors in your code

### Tools not appearing
- Ensure tools are registered in `register_tools()`
- Check tool names are unique
- Verify input schema is valid JSON Schema

### Tool execution fails
- Check handler function is async
- Verify parameter names match input schema
- Add logging to debug: `self.logger.info(f"Arguments: {arguments}")`

## 🤝 Contributing

When you create something useful:
1. Add proper documentation
2. Include usage examples
3. Write tests
4. Consider contributing back to the repository

## 📄 License

This template follows the same license as the parent project.
