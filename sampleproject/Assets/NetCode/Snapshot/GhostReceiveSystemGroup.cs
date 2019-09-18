using System.Collections.Generic;
using Unity.Collections;
using Unity.Entities;
using Unity.Transforms;
using Unity.Jobs;

[UpdateInGroup(typeof(ClientSimulationSystemGroup))]
[UpdateAfter(typeof(NetworkStreamReceiveSystem))]
[UpdateBefore(typeof(TransformSystemGroup))]
public class GhostReceiveSystemGroup : ComponentSystemGroup
{
    // having the group own the ghost map is a bit of a hack to solve a problem with accessing the receiver system from the default spawn system (because it is generic)
    protected override void OnCreateManager()
    {
        m_ghostEntityMap = new NativeHashMap<int, GhostEntity>(2048, Allocator.Persistent);
        m_GhostRemoveList = new NativeList<GhostRemove>(512, Allocator.Persistent);
        m_GhostTypeToEntityMap = new Dictionary<int, NativeHashMap<int, GhostEntity>>(16);
    }

    protected override void OnDestroyManager()
    {
        m_ghostEntityMap.Dispose();
        m_GhostRemoveList.Dispose();
        foreach (var m in m_GhostTypeToEntityMap.Values)
        {
            m.Dispose();
        }
    }

    protected override void OnUpdate()
    {
        // m_GhostTypeToEntityMap -> m_ghostEntityMap
        m_GhostEntityMapProducerHandle.Complete();
        m_GhostEntityMapProducerHandle = default;
        m_ghostEntityMap.Clear();
        foreach (var m in m_GhostTypeToEntityMap.Values)
        {
            var ghostIdArray = m.GetKeyArray(Allocator.Temp);
            foreach (var ghostId in ghostIdArray)
            {
                m_ghostEntityMap.TryAdd(ghostId, m[ghostId]);
            }
            ghostIdArray.Dispose();
        }
        
        base.OnUpdate();
    }

    internal NativeHashMap<int, GhostEntity> GetGhostEntityMap(int ghostType)
    {
        if (m_GhostTypeToEntityMap.TryGetValue(ghostType, out var m))
            return m;
        m = new NativeHashMap<int, GhostEntity>(2048, Allocator.Persistent);
        m_GhostTypeToEntityMap.Add(ghostType, m);
        return m;
    }
    
    internal void AddJobHandleForGhostEntityMapProducer(JobHandle producerJob)
    {
        m_GhostEntityMapProducerHandle = JobHandle.CombineDependencies(m_GhostEntityMapProducerHandle, producerJob);
    }
    
    internal void AddJobHandleForGhostRemoveListProducer(JobHandle producerJob)
    {
        m_GhostRemoveListProducerHandle = JobHandle.CombineDependencies(m_GhostRemoveListProducerHandle, producerJob);
    }

    internal void UpdateGhostMapRemove()
    {
        m_GhostRemoveListProducerHandle.Complete();
        m_GhostRemoveListProducerHandle = default;
        {
            for (int i = 0, e = m_GhostRemoveList.Length; i < e; ++i)
            {
                var ghostRemove = m_GhostRemoveList[i];
                m_GhostTypeToEntityMap[ghostRemove.GhostType].Remove(ghostRemove.GhostId);
            }
            m_GhostRemoveList.Clear();
        }
    }

    internal struct GhostRemove
    {
        public int GhostType;
        public int GhostId;
    }

    internal NativeHashMap<int, GhostEntity> GhostEntityMap => m_ghostEntityMap;
    internal NativeList<GhostRemove> GhostRemoveList => m_GhostRemoveList;
    
    private NativeHashMap<int, GhostEntity> m_ghostEntityMap;
    private Dictionary<int, NativeHashMap<int, GhostEntity>> m_GhostTypeToEntityMap;
    private NativeList<GhostRemove> m_GhostRemoveList;
    private JobHandle m_GhostEntityMapProducerHandle;
    private JobHandle m_GhostRemoveListProducerHandle;
}

[UpdateInGroup(typeof(GhostReceiveSystemGroup))]
public class GhostUpdateSystemGroup : ComponentSystemGroup
{
}

[DisableAutoCreation]
[UpdateInGroup(typeof(ClientSimulationSystemGroup))]
public class GhostSpawnSystemGroup : ComponentSystemGroup
{}

[DisableAutoCreation]
[UpdateInGroup(typeof(ClientSimulationSystemGroup))]
public class GhostSpawnInitSystemGroup : ComponentSystemGroup
{
    private GhostReceiveSystemGroup m_GhostReceiveSystemGroup;
    protected override void OnUpdate()
    {
        if (m_GhostReceiveSystemGroup == null)
            m_GhostReceiveSystemGroup = World.GetExistingSystem<GhostReceiveSystemGroup>();
        m_GhostReceiveSystemGroup.UpdateGhostMapRemove();
        
        base.OnUpdate();
    }
}
