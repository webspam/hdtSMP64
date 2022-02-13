#pragma once
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <skse64/Serialization.h>

namespace hdt {
	class SerializerBase;

	extern PluginHandle g_PluginHandle;

	extern std::vector<SerializerBase*> g_SerializerList;

	class SerializerBase {
	public:
		SerializerBase() {};
		~SerializerBase() {};
		virtual UInt32 StorageName() = 0;
		virtual UInt32 FormatVersion() = 0;

		virtual void SaveData(SKSESerializationInterface*) = 0;

		virtual void ReadData(SKSESerializationInterface*, UInt32) = 0;

		static inline std::vector<SerializerBase*>& GetSerializerList() { return g_SerializerList; };

		static void Save(SKSESerializationInterface* intfc) {
			for (auto data_block : g_SerializerList) {
				data_block->SaveData(intfc);
			}
		};

		static void Load(SKSESerializationInterface* intfc) {
			UInt32 type, version, length;
			//auto load_begin = clock();
			while (intfc->GetNextRecordInfo(&type, &version, &length)) {
				auto record = std::find_if(g_SerializerList.begin(), g_SerializerList.end(), [type, version](SerializerBase* a_srlzr) {
					return type == a_srlzr->StorageName() && version == a_srlzr->FormatVersion();
					}
				);
				if (record == g_SerializerList.end())continue;

				(*record)->ReadData(intfc, length);
			}
			//Less than a microsecond
			//Console_Print("[HDT-SMP] Serializer loading cost: %.3f sec.", (clock() - load_begin) / 1000.0f);
		};
	};

	template<class _Storage_t = void, class _Stream_t = std::stringstream>
	class Serializer :public SerializerBase {
	public:
		Serializer() {
			g_SerializerList.push_back(this);
		};

		~Serializer() {};

		virtual _Stream_t Serialize() = 0;
		virtual _Storage_t Deserialize(_Stream_t&) = 0;

		void SaveData(SKSESerializationInterface*) override;

		void ReadData(SKSESerializationInterface*, UInt32) override;

	protected:
		static inline std::string _toString(_Stream_t& _stream) {
			return _stream.rdbuf()->str();
		};
	};
	
	template<class _Storage_t, class _Stream_t>
	inline void Serializer<_Storage_t, _Stream_t>::SaveData(SKSESerializationInterface* intfc)
	{
		_Stream_t s_data_block = this->Serialize();
		intfc->OpenRecord(this->StorageName(), this->FormatVersion());
		intfc->WriteRecordData(_toString(s_data_block).c_str(), _toString(s_data_block).length());
	}

	template<class _Storage_t, class _Stream_t>
	inline void Serializer<_Storage_t, _Stream_t>::ReadData(SKSESerializationInterface* intfc, UInt32 length)
	{
		char* data_block = nullptr;
		intfc->ReadRecordData(data_block, length);

		std::string s_data(data_block, length);
		_Stream_t _stream; _stream << s_data;
		this->Deserialize(_stream);
	}
}