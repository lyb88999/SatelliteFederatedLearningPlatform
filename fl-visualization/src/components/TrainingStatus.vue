# src/components/TrainingStatus.vue
<template>
  <div class="training-status">
    <h3>Training Status</h3>
    <div v-if="currentClient">
      <div class="status-item">
        <span>Client ID:</span>
        <span>{{ currentClient.client_id }}</span>
      </div>
      <div class="status-item">
        <span>Round:</span>
        <span>{{ currentClient.round }}</span>
      </div>
      <div class="status-item">
        <span>Loss:</span>
        <span>{{ currentClient.loss.toFixed(4) }}</span>
      </div>
      <div class="status-item">
        <span>Accuracy:</span>
        <span>{{ currentClient.accuracy.toFixed(4) }}</span>
      </div>
      <div class="status-item">
        <span>Training Samples:</span>
        <span>{{ currentClient.training_samples }}</span>
      </div>
    </div>
    <div v-else>
      Waiting for training data...
    </div>
  </div>
</template>

<script setup>
import { defineProps } from 'vue'

defineProps({
  currentClient: {
    type: Object,
    default: null
  }
})
</script>

<style scoped>
.training-status {
  position: absolute;
  top: 20px;
  left: 20px;
  background: rgba(0, 0, 0, 0.7);
  color: white;
  padding: 20px;
  border-radius: 8px;
  font-family: Arial, sans-serif;
}

.status-item {
  display: flex;
  justify-content: space-between;
  margin: 5px 0;
}

.status-item span:first-child {
  margin-right: 20px;
  font-weight: bold;
}
</style>

# src/components/ThreeScene.vue
<template>
  <div class="scene-container" ref="container"></div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'
import * as THREE from 'three'

const props = defineProps({
  clients: {
    type: Object,
    required: true
  }
})

const container = ref(null)
let scene, camera, renderer, serverNode
const clientNodes = new Map()
const clientConnections = new Map()

// 初始化Three.js场景
const initScene = () => {
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)
  
  camera = new THREE.PerspectiveCamera(
    75,
    container.value.clientWidth / container.value.clientHeight,
    0.1,
    1000
  )
  camera.position.z = 15
  
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(container.value.clientWidth, container.value.clientHeight)
  container.value.appendChild(renderer.domElement)
  
  // 创建服务器节点
  const serverGeometry = new THREE.SphereGeometry(1, 32, 32)
  const serverMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 })
  serverNode = new THREE.Mesh(serverGeometry, serverMaterial)
  scene.add(serverNode)
  
  // 添加光源
  const light = new THREE.PointLight(0xffffff, 1, 100)
  light.position.set(0, 0, 10)
  scene.add(light)
  scene.add(new THREE.AmbientLight(0x404040))
  
  animate()
}

// 创建客户端节点
const createClientNode = (clientId, position) => {
  const geometry = new THREE.SphereGeometry(0.5, 32, 32)
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 })
  const node = new THREE.Mesh(geometry, material)
  node.position.set(position.x, position.y, position.z)
  scene.add(node)
  
  // 创建连接线
  const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff })
  const points = [
    new THREE.Vector3(0, 0, 0),
    node.position
  ]
  const lineGeometry = new THREE.BufferGeometry().setFromPoints(points)
  const line = new THREE.Line(lineGeometry, lineMaterial)
  scene.add(line)
  
  clientNodes.set(clientId, node)
  clientConnections.set(clientId, line)
}

// 更新客户端状态
const updateClientStatus = (clientId, status) => {
  let node = clientNodes.get(clientId)
  if (!node) {
    const angle = clientNodes.size * (Math.PI * 2) / 8
    const position = new THREE.Vector3(
      Math.cos(angle) * 5,
      Math.sin(angle) * 5,
      0
    )
    createClientNode(clientId, position)
    node = clientNodes.get(clientId)
  }

  // 更新节点颜色
  const material = node.material
  const loss = status.loss || 0
  const hue = Math.max(0, Math.min(1, 1 - loss / 2))
  material.color.setHSL(hue, 1, 0.5)
}

// 动画循环
const animate = () => {
  requestAnimationFrame(animate)
  
  clientNodes.forEach((node) => {
    node.rotation.x += 0.01
    node.rotation.y += 0.01
  })
  
  renderer.render(scene, camera)
}

// 监听窗口大小变化
const handleResize = () => {
  if (camera && renderer && container.value) {
    camera.aspect = container.value.clientWidth / container.value.clientHeight
    camera.updateProjectionMatrix()
    renderer.setSize(container.value.clientWidth, container.value.clientHeight)
  }
}

// 监听clients属性变化
watch(() => props.clients, (newClients) => {
  if (newClients) {
    Object.entries(newClients).forEach(([clientId, status]) => {
      updateClientStatus(clientId, status)
    })
  }
}, { deep: true })

onMounted(() => {
  initScene()
  window.addEventListener('resize', handleResize)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', handleResize)
})
</script>

<style scoped>
.scene-container {
  width: 100vw;
  height: 100vh;
}
</style>

# src/App.vue
<template>
  <div class="app">
    <ThreeScene :clients="clients" />
    <TrainingStatus :current-client="currentClient" />
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import ThreeScene from './components/ThreeScene.vue'
import TrainingStatus from './components/TrainingStatus.vue'

const clients = ref({})
const currentClient = ref(null)
let ws = null

const connectWebSocket = () => {
  ws = new WebSocket('ws://localhost:8765')
  
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    clients.value = data.clients
    
    // 更新当前选中的客户端
    if (Object.keys(data.clients).length > 0) {
      currentClient.value = data.clients[Object.keys(data.clients)[0]]
    }
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
  }
  
  ws.onclose = () => {
    setTimeout(connectWebSocket, 5000) // 断线重连
  }
}

onMounted(() => {
  connectWebSocket()
})

onBeforeUnmount(() => {
  if (ws) {
    ws.close()
  }
})
</script>

<style>
.app {
  width: 100vw;
  height: 100vh;
  overflow: hidden;
}
</style>
