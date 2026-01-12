# Node-RED Service

This directory acts as the workspace for the Node-RED service, which is orchestrating data flow and logic for the Sack Detector Edge project.

This README focuses on the setup and configuration of the Node-RED service itself. For details on the overall system workflows (Sack Counting, Log Collection), please refer to the main project [README](../README.md) or the [Detector Documentation](../detector/README.md).

## Directory Structure

```text
nodered/
├── data/               # Persistent data directory mapped to /data in the container
│   ├── settings.js     # Node-RED settings file
│   ├── flows.json      # Flow configuration
│   ├── package.json    # User-installed dependencies
│   └── node_modules/   # Installed node modules
└── README.md           # This documentation
```

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed.
- [Docker Compose](https://docs.docker.com/compose/install/) installed.

### Running the Service

The Node-RED service is defined in the root `docker-compose.yml` file. To start it alongside other services, run:

```bash
docker-compose up -d nodered
```

*Note: Node-RED depends on the `mqtt` and `sqlite` services, which will start automatically if not already running.*

### Accessing the Interface

Once the container is running, locally you can access the Node-RED editor in your browser:

- **URL**: [http://localhost:1880](http://localhost:1880)

## Configuration

### Environment & Ports

- **Port**: `1880` is exposed to the host system.
- **Image**: `nodered/node-red:latest`

### Volumes

The service mounts several local directories to persist data and access shared resources:

| Host Path | Container Path | Description |
|-----------|----------------|-------------|
| `./nodered/data` | `/data` | Stores user directory, flows, nodes, and settings. |
| `./sqlite/data` | `/data/sqlite` | Provides access to the shared SQLite database. |
| `./detector/records` | `/data/records` | Access to detector logs for collection. |

### Environment Variables

The service uses the following environment variables (defined in `docker-compose.yml`):

- `EXTERNAL_MQTT_HOST`
- `EXTERNAL_MQTT_PORT`
- `EXTERNAL_MQTT_USER`
- `EXTERNAL_MQTT_PASS`

These are used within the flow to connect to external services.

## Persistence & Git

The `data/` directory is **tracked by Git**. This means:

1.  **Flows are Persisted**: Your `flows.json` and credentials are saved in the repository. When you push/pull to another device, the same flows will be deployed.
2.  **Sensitive Data**: Be careful not to commit sensitive passwords in `flows_cred.json` or `settings.js` if this is a public repository. Use environment variables where possible.

