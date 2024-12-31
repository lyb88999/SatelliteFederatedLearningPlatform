<!-- src/App.vue -->
<template>
  <div class="app">
    <div class="scene-container" ref="sceneContainer"></div>
    <div class="status-panel">
      <h3>Training Status</h3>
      <div v-if="Object.keys(clients).length > 0">
        <div v-for="(client, id) in clients" :key="id" class="client-status">
          <h4>Client {{ id }}</h4>
          <div>Round: {{ client.round }}</div>
          <div>Loss: {{ client.loss?.toFixed(4) }}</div>
          <div>Accuracy: {{ client.accuracy?.toFixed(4) }}</div>
          <div>Samples: {{ client.training_samples }}</div>
        </div>
      </div>
      <div v-else>
        Waiting for clients...
      </div>
    </div>
  </div>
</template>

<script>
import * as THREE from 'three';
import { ref, onMounted, onBeforeUnmount } from 'vue';

export default {
  name: 'App',
  setup() {
    const sceneContainer = ref(null);
    const clients = ref({});
    let scene, camera, renderer;
    const clientNodes = new Map();
    const clientConnections = new Map();

    // 初始化Three.js场景
    const initScene = () => {
      scene = new THREE.Scene();
      scene.background = new THREE.Color(0x000000);

      camera = new THREE.PerspectiveCamera(
        75,
        window.innerWidth / window.innerHeight,
        0.1,
        1000
      );
      camera.position.z = 15;

      renderer = new THREE.WebGLRenderer({ antialias: true });
      renderer.setSize(window.innerWidth, window.innerHeight);
      sceneContainer.value.appendChild(renderer.domElement);

      // 添加中心服务器节点
      const serverGeometry = new THREE.SphereGeometry(1, 32, 32);
      const serverMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 });
      const serverNode = new THREE.Mesh(serverGeometry, serverMaterial);
      scene.add(serverNode);

      // 添加光源
      const light = new THREE.PointLight(0xffffff, 1, 100);
      light.position.set(0, 0, 10);
      scene.add(light);
      scene.add(new THREE.AmbientLight(0x404040));

      animate();
    };

    // 创建客户端节点
    const createClientNode = (clientId, position) => {
      const geometry = new THREE.SphereGeometry(0.5, 32, 32);
      const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
      const node = new THREE.Mesh(geometry, material);
      node.position.set(position.x, position.y, position.z);
      scene.add(node);

      // 创建连接线
      const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
      const points = [
        new THREE.Vector3(0, 0, 0),
        node.position
      ];
      const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
      const line = new THREE.Line(lineGeometry, lineMaterial);
      scene.add(line);

      clientNodes.set(clientId, node);
      clientConnections.set(clientId, line);
    };

    // 更新客户端节点
    const updateClientNode = (clientId, status) => {
      let node = clientNodes.get(clientId);
      if (!node) {
        const angle = clientNodes.size * (Math.PI * 2) / 8;
        const position = new THREE.Vector3(
          Math.cos(angle) * 5,
          Math.sin(angle) * 5,
          0
        );
        createClientNode(clientId, position);
        node = clientNodes.get(clientId);
      }

      // 根据loss更新颜色
      const material = node.material;
      const loss = status.loss || 0;
      const hue = Math.max(0, Math.min(1, 1 - loss / 2));
      material.color.setHSL(hue, 1, 0.5);
    };

    // 动画循环
    const animate = () => {
      requestAnimationFrame(animate);
      
      // 旋转节点
      clientNodes.forEach((node) => {
        node.rotation.x += 0.01;
        node.rotation.y += 0.01;
      });

      renderer.render(scene, camera);
    };

    // WebSocket连接
    const connectWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8765');

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        clients.value = data.clients;
        
        // 更新3D场景中的节点
        Object.entries(data.clients).forEach(([clientId, status]) => {
          updateClientNode(clientId, status);
        });
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      return ws;
    };

    // 生命周期钩子
    onMounted(() => {
      initScene();
      const ws = connectWebSocket();

      // 清理函数
      onBeforeUnmount(() => {
        if (ws) ws.close();
        if (renderer) {
          renderer.dispose();
          sceneContainer.value?.removeChild(renderer.domElement);
        }
      });
    });

    return {
      sceneContainer,
      clients
    };
  }
};
</script>

<style>
.app {
  width: 100vw;
  height: 100vh;
  margin: 0;
  padding: 0;
  overflow: hidden;
  background-color: #000;
}

.scene-container {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
}

.status-panel {
  position: absolute;
  top: 20px;
  left: 20px;
  padding: 20px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  border-radius: 8px;
  font-family: Arial, sans-serif;
  z-index: 1000;
}

.client-status {
  margin-bottom: 15px;
  padding: 10px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}

.client-status h4 {
  margin: 0 0 10px 0;
}
</style>
