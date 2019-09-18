
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Networking.Transport.Utilities;
using UnityEngine;

[UpdateInGroup(typeof(GhostSpawnSystemGroup))]
[AlwaysUpdateSystem]
public abstract class DefaultGhostSpawnSystem<T> : JobComponentSystem
    where T: struct, ISnapshotData<T>
{
    public int GhostType { get; set; }
    public NativeList<T> NewGhosts => m_NewGhosts;
    public NativeList<int> NewGhostIds => m_NewGhostIds;

    public NativeQueue<T> SpawnRequestQueue => m_SpawnRequestQueue;

    public void SetSpawnRequestQueueProducer(JobHandle producerHandle)
    {
        if (!m_SpawnRequestQueueProducerHandle.IsCompleted)
        {
            Debug.LogWarning("m_SpawnRequestQueueProducerHandle set multiple times in one update loop");
            m_SpawnRequestQueueProducerHandle.Complete();
        }

        m_SpawnRequestQueueProducerHandle = producerHandle;
    }
    
    
    private NativeList<T> m_NewGhosts;
    private NativeList<int> m_NewGhostIds;
    private EntityArchetype m_Archetype;
    private EntityArchetype m_PredictedArchetype;
    private EntityArchetype m_InitialArchetype;
    private NativeHashMap<int, GhostEntity> m_GhostMap;
    private NativeHashMap<int, GhostEntity>.Concurrent m_ConcurrentGhostMap;
    private EntityQuery m_DestroyGroup;
    private JobHandle m_SpawnRequestQueueProducerHandle;
    private NativeQueue<T> m_SpawnRequestQueue;
    private EntityQuery m_PrefabGroup;

    private NativeList<Entity> m_InvalidGhosts;

    struct DelayedSpawnGhost
    {
        public int ghostId;
        public uint spawnTick;
        public Entity oldEntity;
    }

    public struct PredictSpawnGhost
    {
        public T snapshotData;
        public Entity entity;
    }
    private NativeList<PredictSpawnGhost> m_PredictSpawnGhosts;
    private NativeHashMap<int, int> m_PredictionSpawnCleanupMap;

    private NativeQueue<DelayedSpawnGhost> m_DelayedSpawnQueue;
    private NativeQueue<DelayedSpawnGhost>.Concurrent m_ConcurrentDelayedSpawnQueue;
    private NativeList<DelayedSpawnGhost> m_CurrentDelayedSpawnList;
    private NativeQueue<DelayedSpawnGhost> m_PredictedSpawnQueue;
    private NativeQueue<DelayedSpawnGhost>.Concurrent m_ConcurrentPredictedSpawnQueue;
    // The entities which need to wait to be spawned on the right tick (interpolated)
    private NativeList<DelayedSpawnGhost> m_CurrentPredictedSpawnList;
    private EndSimulationEntityCommandBufferSystem m_Barrier;
    private NetworkTimeSystem m_TimeSystem;

    private GhostSpawnInitSystem m_GhostSpawnInitSystem;
    private GhostReceiveSystemGroup m_GhostReceiveSystemGroup;

    protected abstract EntityArchetype GetGhostArchetype();
    protected abstract EntityArchetype GetPredictedGhostArchetype();

    protected virtual JobHandle UpdateNewInterpolatedEntities(NativeArray<Entity> entities, JobHandle inputDeps)
    {
        return inputDeps;
    }
    protected virtual JobHandle UpdateNewPredictedEntities(NativeArray<Entity> entities, JobHandle inputDeps)
    {
        return inputDeps;
    }

    protected virtual JobHandle MarkPredictedGhosts(NativeArray<T> snapshots, NativeArray<int> predictionMask,
        NativeList<PredictSpawnGhost> predictSpawnGhosts, JobHandle inputDeps)
    {
        return inputDeps;
    }

    protected override void OnCreateManager()
    {
        m_NewGhosts = new NativeList<T>(16, Allocator.Persistent);
        m_NewGhostIds = new NativeList<int>(16, Allocator.Persistent);
        m_Archetype = GetGhostArchetype();
        m_PredictedArchetype = GetPredictedGhostArchetype();
        m_InitialArchetype = EntityManager.CreateArchetype(ComponentType.ReadWrite<T>(), ComponentType.ReadWrite<ReplicatedEntityComponent>());
        
        m_DestroyGroup = GetEntityQuery(ComponentType.ReadOnly<T>(),
            ComponentType.Exclude<ReplicatedEntityComponent>(), ComponentType.Exclude<PredictedSpawnRequestComponent>(),
            ComponentType.Exclude<GhostClientPrefabComponent>());
        m_SpawnRequestQueue = new NativeQueue<T>(Allocator.Persistent);
        m_PrefabGroup = GetEntityQuery(ComponentType.ReadOnly<T>(),
            ComponentType.ReadOnly<GhostClientPrefabComponent>());

        m_InvalidGhosts = new NativeList<Entity>(1024, Allocator.Persistent);
        m_DelayedSpawnQueue = new NativeQueue<DelayedSpawnGhost>(Allocator.Persistent);
        m_CurrentDelayedSpawnList = new NativeList<DelayedSpawnGhost>(1024, Allocator.Persistent);
        m_ConcurrentDelayedSpawnQueue = m_DelayedSpawnQueue.ToConcurrent();
        m_PredictedSpawnQueue = new NativeQueue<DelayedSpawnGhost>(Allocator.Persistent);
        m_CurrentPredictedSpawnList = new NativeList<DelayedSpawnGhost>(1024, Allocator.Persistent);
        m_ConcurrentPredictedSpawnQueue = m_PredictedSpawnQueue.ToConcurrent();
        m_Barrier = World.GetOrCreateSystem<EndSimulationEntityCommandBufferSystem>();

        m_PredictSpawnGhosts = new NativeList<PredictSpawnGhost>(16, Allocator.Persistent);
        m_PredictionSpawnCleanupMap = new NativeHashMap<int, int>(16, Allocator.Persistent);

        m_GhostSpawnInitSystem = World.CreateSystem<GhostSpawnInitSystem>(this);
        World.GetOrCreateSystem<GhostSpawnInitSystemGroup>().AddSystemToUpdateList(m_GhostSpawnInitSystem);
        m_GhostReceiveSystemGroup = World.GetExistingSystem<GhostReceiveSystemGroup>();
        m_TimeSystem = World.GetOrCreateSystem<NetworkTimeSystem>();
    }

    protected override void OnDestroyManager()
    {
        m_NewGhosts.Dispose();
        m_NewGhostIds.Dispose();
        
        m_SpawnRequestQueue.Dispose();
        
        m_InvalidGhosts.Dispose();
        m_DelayedSpawnQueue.Dispose();
        m_CurrentDelayedSpawnList.Dispose();
        m_PredictedSpawnQueue.Dispose();
        m_CurrentPredictedSpawnList.Dispose();

        m_PredictSpawnGhosts.Dispose();
        m_PredictionSpawnCleanupMap.Dispose();
    }

    [BurstCompile]
    struct CopyInitialStateJob : IJobParallelFor
    {
        [DeallocateOnJobCompletion] [ReadOnly] public NativeArray<Entity> entities;
        [ReadOnly] public NativeList<T> newGhosts;
        [ReadOnly] public NativeList<int> newGhostIds;
        [NativeDisableParallelForRestriction] public BufferFromEntity<T> snapshotFromEntity;
        public NativeHashMap<int, GhostEntity>.Concurrent ghostMap;
        public int ghostType;
        public NativeQueue<DelayedSpawnGhost>.Concurrent pendingSpawnQueue;
        public NativeQueue<DelayedSpawnGhost>.Concurrent predictedSpawnQueue;
        [DeallocateOnJobCompletion][ReadOnly] public NativeArray<int> predictionMask;
        [ReadOnly] public NativeList<PredictSpawnGhost> predictionSpawnGhosts;
        public NativeHashMap<int, int>.Concurrent predictionSpawnCleanupMap;
        public EntityCommandBuffer.Concurrent commandBuffer;
        public void Execute(int i)
        {
            var entity = entities[i];
            if (predictionMask[i] == 0)
            {
                pendingSpawnQueue.Enqueue(new DelayedSpawnGhost{ghostId = newGhostIds[i], spawnTick = newGhosts[i].Tick, oldEntity = entity});
            }
            // If multiple entities map to the same prediction spawned entity, the first one will get it, the others are treated like regular spawns
            else if (predictionMask[i] > 1 && predictionSpawnCleanupMap.TryAdd(predictionMask[i]-2, 1))
            {
                commandBuffer.DestroyEntity(i, entity);
                entity = predictionSpawnGhosts[predictionMask[i]-2].entity;
            }
            else
            {
                predictedSpawnQueue.Enqueue(new DelayedSpawnGhost
                    {ghostId = newGhostIds[i], spawnTick = newGhosts[i].Tick, oldEntity = entity});
            }
            var snapshot = snapshotFromEntity[entity];
            snapshot.ResizeUninitialized(1);
            snapshot[0] = newGhosts[i];
            ghostMap.TryAdd(newGhostIds[i], new GhostEntity
            {
                entity = entity,
                ghostType = ghostType
            });
        }
    }

    [BurstCompile]
    struct DelayedSpawnJob : IJob
    {
        [ReadOnly] public NativeArray<Entity> entities;
        [ReadOnly] public NativeList<DelayedSpawnGhost> delayedGhost;
        [NativeDisableParallelForRestriction] public BufferFromEntity<T> snapshotFromEntity;
        public NativeHashMap<int, GhostEntity> ghostMap;
        public int ghostType;
        public void Execute()
        {
            for (int i = 0; i < entities.Length; ++i)
            {
                var newSnapshot = snapshotFromEntity[entities[i]];
                var oldSnapshot = snapshotFromEntity[delayedGhost[i].oldEntity];
                newSnapshot.ResizeUninitialized(oldSnapshot.Length);
                for (int snap = 0; snap < newSnapshot.Length; ++snap)
                    newSnapshot[snap] = oldSnapshot[snap];
                ghostMap.Remove(delayedGhost[i].ghostId);
                ghostMap.TryAdd(delayedGhost[i].ghostId, new GhostEntity
                {
                    entity = entities[i],
                    ghostType = ghostType
                });
            }
        }
    }

    [BurstCompile]
    struct DeallocateJob : IJob
    {
        [DeallocateOnJobCompletion] [ReadOnly] public NativeArray<Entity> array;
        public void Execute()
        {
        }
    }

    [BurstCompile]
    struct ClearNewJob : IJob
    {
        public NativeList<T> newGhosts;
        public NativeList<int> newGhostIds;
        public void Execute()
        {
            newGhosts.Clear();
            newGhostIds.Clear();
        }
    }

    [BurstCompile]
    struct PredictSpawnCleanupJob : IJob
    {
        public NativeHashMap<int, int> predictionSpawnCleanupMap;
        public NativeList<PredictSpawnGhost> predictionSpawnGhosts;
        public uint interpolationTarget;
        public EntityCommandBuffer commandBuffer;
        public ComponentType replicatedEntityComponentType;
        public void Execute()
        {
            var keys = predictionSpawnCleanupMap.GetKeyArray(Allocator.Temp);
            for (var i = 0; i < keys.Length; ++i)
                predictionSpawnGhosts[keys[i]] = default(PredictSpawnGhost);
            for (int i = 0; i < predictionSpawnGhosts.Length; ++i)
            {
                if (predictionSpawnGhosts[i].entity != Entity.Null &&
                    SequenceHelpers.IsNewer(interpolationTarget, predictionSpawnGhosts[i].snapshotData.Tick))
                {
                    // Trigger a delete of the entity
                    commandBuffer.RemoveComponent(predictionSpawnGhosts[i].entity, replicatedEntityComponentType);
                    predictionSpawnGhosts[i] = default(PredictSpawnGhost);
                }
                if (predictionSpawnGhosts[i].entity == Entity.Null)
                {
                    predictionSpawnGhosts.RemoveAtSwapBack(i);
                    --i;
                }
            }
        }
    }

    private class GhostSpawnInitSystem: ComponentSystem
    {
        private readonly DefaultGhostSpawnSystem<T> m_GhostSpawnSystem;
        public NativeArray<Entity> delayedEntities;
        public NativeArray<Entity> predictedEntities;
        public NativeArray<Entity> predictSpawnEntities;
        public NativeArray<Entity> newEntities;
        public JobHandle newGhostsChangeHandle;

        public GhostSpawnInitSystem(DefaultGhostSpawnSystem<T> ghostSpawnSystem)
        {
            m_GhostSpawnSystem = ghostSpawnSystem;
        }

        protected override void OnUpdate()
        {
            if (!m_GhostSpawnSystem.m_GhostMap.IsCreated)
            {
                //Debug.LogFormat("GhostMap initialize {0} {1}", GetType(), m_GhostSpawnSystem.GhostType);
                m_GhostSpawnSystem.m_GhostMap = m_GhostSpawnSystem.m_GhostReceiveSystemGroup
                    .GetGhostEntityMap(m_GhostSpawnSystem.GhostType);
                m_GhostSpawnSystem.m_ConcurrentGhostMap = m_GhostSpawnSystem.m_GhostMap.ToConcurrent();
            }

            if (!m_GhostSpawnSystem.m_DestroyGroup.IsEmptyIgnoreFilter)
            {
                EntityManager.DestroyEntity(m_GhostSpawnSystem.m_DestroyGroup);
            }
                
            if (m_GhostSpawnSystem.m_InvalidGhosts.Length > 0)
            {
                EntityManager.DestroyEntity(m_GhostSpawnSystem.m_InvalidGhosts);
                m_GhostSpawnSystem.m_InvalidGhosts.Clear();
            }
                

            var targetTick = NetworkTimeSystem.interpolateTargetTick;
            m_GhostSpawnSystem.m_CurrentDelayedSpawnList.Clear();
            while (m_GhostSpawnSystem.m_DelayedSpawnQueue.Count > 0 &&
                   !SequenceHelpers.IsNewer(m_GhostSpawnSystem.m_DelayedSpawnQueue.Peek().spawnTick, targetTick))
            {
                var ghost = m_GhostSpawnSystem.m_DelayedSpawnQueue.Dequeue();
                GhostEntity gent;
                if (m_GhostSpawnSystem.m_GhostMap.TryGetValue(ghost.ghostId, out gent))
                {
                    m_GhostSpawnSystem.m_CurrentDelayedSpawnList.Add(ghost);
                    m_GhostSpawnSystem.m_InvalidGhosts.Add(gent.entity);
                }
            }
            m_GhostSpawnSystem.m_CurrentPredictedSpawnList.Clear();
            while (m_GhostSpawnSystem.m_PredictedSpawnQueue.Count > 0)
            {
                var ghost = m_GhostSpawnSystem.m_PredictedSpawnQueue.Dequeue();
                GhostEntity gent;
                if (m_GhostSpawnSystem.m_GhostMap.TryGetValue(ghost.ghostId, out gent))
                {
                    m_GhostSpawnSystem.m_CurrentPredictedSpawnList.Add(ghost);
                    m_GhostSpawnSystem.m_InvalidGhosts.Add(gent.entity);
                }
            }

            var prefabs = m_PrefabGroup.ToComponentDataArray<GhostClientPrefabComponent>(Allocator.TempJob);
            
            if (m_GhostSpawnSystem.m_CurrentDelayedSpawnList.Length > 0)
            {
                delayedEntities = new NativeArray<Entity>(
                    m_GhostSpawnSystem.m_CurrentDelayedSpawnList.Length,
                    Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
				if (prefabs.Length == 1)
                	EntityManager.Instantiate(prefabs[0].interpolatedPrefab, delayedEntities);
            	else
                	EntityManager.CreateEntity(m_GhostSpawnSystem.m_Archetype, delayedEntities);
            }
            
            if (m_GhostSpawnSystem.m_CurrentPredictedSpawnList.Length > 0)
            {
                predictedEntities = new NativeArray<Entity>(
                    m_GhostSpawnSystem.m_CurrentPredictedSpawnList.Length,
                    Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
				if (prefabs.Length == 1)
                	EntityManager.Instantiate(prefabs[0].predictedPrefab, predictedEntities);
            	else
                	EntityManager.CreateEntity(m_GhostSpawnSystem.m_PredictedArchetype, predictedEntities);
            }

            {
                m_GhostSpawnSystem.m_SpawnRequestQueueProducerHandle.Complete();
                m_GhostSpawnSystem.m_SpawnRequestQueueProducerHandle = default;
                var spawnRequestQueue = m_GhostSpawnSystem.m_SpawnRequestQueue;
                if (spawnRequestQueue.Count > 0)
                {
                    predictSpawnEntities = new NativeArray<Entity>(
                        spawnRequestQueue.Count, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                    var i = 0;
                    while (spawnRequestQueue.Count > 0)
                    {
                        var spawnRequest = spawnRequestQueue.Dequeue();
                        Entity entity;
            			if (prefabs.Length == 1)
                			entity = EntityManager.Instantiate(prefabs[0].predictedPrefab);
                		else
                        	entity = EntityManager.CreateEntity(m_GhostSpawnSystem.m_PredictedArchetype);
                        var buffer = EntityManager.GetBuffer<T>(entity);
                        buffer.ResizeUninitialized(1);
                        buffer[0] = spawnRequest;
                        predictSpawnEntities[i++] = entity;
                        m_GhostSpawnSystem.m_PredictSpawnGhosts.Add(new PredictSpawnGhost {snapshotData = spawnRequest, entity = entity});
                    }
                }
                else
                {
                    predictSpawnEntities = default;
                }
            }

            prefabs.Dispose();

            newGhostsChangeHandle.Complete();
            newGhostsChangeHandle = default;
            
            if (m_GhostSpawnSystem.m_NewGhosts.Length > 0)
            {
                newEntities = new NativeArray<Entity>(
                    m_GhostSpawnSystem.m_NewGhosts.Length, Allocator.TempJob, NativeArrayOptions.UninitializedMemory);
                EntityManager.CreateEntity(m_GhostSpawnSystem.m_InitialArchetype, newEntities);
            }
        }
    }

    protected override JobHandle OnUpdate(JobHandle inputDeps)
    {
        if (m_CurrentDelayedSpawnList.Length > 0)
        {
            var delayedjob = new DelayedSpawnJob
            {
                entities = m_GhostSpawnInitSystem.delayedEntities,
                delayedGhost = m_CurrentDelayedSpawnList,
                snapshotFromEntity = GetBufferFromEntity<T>(),
                ghostMap = m_GhostMap,
                ghostType = GhostType
            };
            inputDeps = delayedjob.Schedule(inputDeps);
            m_GhostReceiveSystemGroup.AddJobHandleForGhostEntityMapProducer(inputDeps);
            inputDeps = UpdateNewInterpolatedEntities(m_GhostSpawnInitSystem.delayedEntities, inputDeps);
            new DeallocateJob {array = m_GhostSpawnInitSystem.delayedEntities}.Schedule(inputDeps);
        }
        // FIXME: current and predicted can run in parallel I think
        if (m_CurrentPredictedSpawnList.Length > 0)
        {
            var delayedjob = new DelayedSpawnJob
            {
                entities = m_GhostSpawnInitSystem.predictedEntities,
                delayedGhost = m_CurrentPredictedSpawnList,
                snapshotFromEntity = GetBufferFromEntity<T>(),
                ghostMap = m_GhostMap,
                ghostType = GhostType
            };
            inputDeps = delayedjob.Schedule(inputDeps);
            m_GhostReceiveSystemGroup.AddJobHandleForGhostEntityMapProducer(inputDeps);
            inputDeps = UpdateNewPredictedEntities(m_GhostSpawnInitSystem.predictedEntities, inputDeps);
            new DeallocateJob {array = m_GhostSpawnInitSystem.predictedEntities}.Schedule(inputDeps);
        }
        if (m_GhostSpawnInitSystem.predictSpawnEntities.IsCreated)
        {
            inputDeps = UpdateNewPredictedEntities(m_GhostSpawnInitSystem.predictSpawnEntities, inputDeps);
            new DeallocateJob {array = m_GhostSpawnInitSystem.predictSpawnEntities}.Schedule(inputDeps);
        }

        m_PredictionSpawnCleanupMap.Clear();
        if (m_NewGhosts.Length > 0)
        {
            if (m_PredictionSpawnCleanupMap.Capacity < m_NewGhosts.Length)
                m_PredictionSpawnCleanupMap.Capacity = m_NewGhosts.Length;
            var predictionMask = new NativeArray<int>(m_NewGhosts.Length, Allocator.TempJob);
            inputDeps = MarkPredictedGhosts(m_NewGhosts, predictionMask, m_PredictSpawnGhosts, inputDeps);
            var job = new CopyInitialStateJob
            {
                entities = m_GhostSpawnInitSystem.newEntities,
                newGhosts = m_NewGhosts,
                newGhostIds = m_NewGhostIds,
                snapshotFromEntity = GetBufferFromEntity<T>(),
                ghostMap = m_ConcurrentGhostMap,
                ghostType = GhostType,
                pendingSpawnQueue = m_ConcurrentDelayedSpawnQueue,
                predictedSpawnQueue = m_ConcurrentPredictedSpawnQueue,
                predictionMask = predictionMask,
                predictionSpawnGhosts = m_PredictSpawnGhosts,
                predictionSpawnCleanupMap = m_PredictionSpawnCleanupMap.ToConcurrent(),
                commandBuffer = m_Barrier.CreateCommandBuffer().ToConcurrent()
            };
            inputDeps = job.Schedule(m_GhostSpawnInitSystem.newEntities.Length, 8, inputDeps);
            m_GhostReceiveSystemGroup.AddJobHandleForGhostEntityMapProducer(inputDeps);
            m_Barrier.AddJobHandleForProducer(inputDeps);
        }

        var targetTick = m_TimeSystem.interpolateTargetTick;
        var spawnClearJob = new PredictSpawnCleanupJob
        {
            predictionSpawnCleanupMap = m_PredictionSpawnCleanupMap,
            predictionSpawnGhosts = m_PredictSpawnGhosts,
            interpolationTarget = targetTick,
            commandBuffer = m_Barrier.CreateCommandBuffer(),
            replicatedEntityComponentType = ComponentType.ReadWrite<ReplicatedEntityComponent>()
        };
        inputDeps = spawnClearJob.Schedule(inputDeps);
        m_Barrier.AddJobHandleForProducer(inputDeps);

        if (m_NewGhosts.Length > 0 || m_NewGhostIds.Length > 0)
        {
            var clearJob = new ClearNewJob
            {
                newGhosts = m_NewGhosts,
                newGhostIds = m_NewGhostIds,
            };
            inputDeps = clearJob.Schedule(inputDeps);
            m_GhostSpawnInitSystem.newGhostsChangeHandle = inputDeps;
        }
        
        return inputDeps;
    }
}

