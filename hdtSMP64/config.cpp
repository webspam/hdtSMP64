#include "config.h"
#include "XmlReader.h"

#include "hdtSkyrimPhysicsWorld.h"

#include "../hdtSSEUtils/LogUtils.h"

#include <clocale>

namespace hdt
{
	static void solver(XMLReader& reader)
	{
		while (reader.Inspect())
		{
			switch (reader.GetInspected())
			{
			case XMLReader::Inspected::StartTag:
				if (reader.GetLocalName() == "numIterations")
					SkyrimPhysicsWorld::get()->getSolverInfo().m_numIterations = btClamped(reader.readInt(), 4, 128);
				else if (reader.GetLocalName() == "groupIterations")
					ConstraintGroup::MaxIterations = btClamped(reader.readInt(), 0, 4096);
				else if (reader.GetLocalName() == "groupEnableMLCP")
					ConstraintGroup::EnableMLCP = reader.readBool();
				else if (reader.GetLocalName() == "erp")
					SkyrimPhysicsWorld::get()->getSolverInfo().m_erp = btClamped(reader.readFloat(), 0.01f, 1.0f);
				else if (reader.GetLocalName() == "min-fps")
					TIME_TICK = 1.0f / (btClamped(reader.readInt(), 1, 300));
				else
				{
					LogWarning("Unknown config : ", reader.GetLocalName());
					reader.skipCurrentElement();
				}
				break;
			case XMLReader::Inspected::EndTag:
				return;
			}
		}
	}

	//static void wind(XMLReader& reader)
	//{
	//	while (reader.Inspect())
	//	{
	//		switch (reader.GetInspected())
	//		{
	//		case XMLReader::Inspected::StartTag:
	//			if (reader.GetLocalName() == "windStrength")
	//				SkyrimPhysicsWorld::get()->getWindCtrl()->m_windStrength = btClamped(reader.readFloat(), 0.f, 10000.f);
	//			else
	//			{
	//				LogWarning("Unknown config : ", reader.GetLocalName());
	//				reader.skipCurrentElement();
	//			}
	//			break;
	//		case XMLReader::Inspected::EndTag:
	//			return;
	//		}
	//	}
	//}

	static void config(XMLReader& reader)
	{
		while (reader.Inspect())
		{
			switch (reader.GetInspected())
			{
			case XMLReader::Inspected::StartTag:
				if (reader.GetLocalName() == "solver")
					solver(reader);
				//else if (reader.GetLocalName() == "wind")
				//	wind(reader);
				else
				{
					LogWarning("Unknown config : ", reader.GetLocalName());
					reader.skipCurrentElement();
				}
				break;
			case XMLReader::Inspected::EndTag:
				return;
			}
		}
	}

	void loadConfig()
	{
		auto bytes = readAllFile2("data/skse/plugins/hdtSkinnedMeshConfigs/configs.xml");
		if (bytes.empty()) return;

		// Store original locale
		char saved_locale[32];
		strcpy_s(saved_locale, std::setlocale(LC_NUMERIC, nullptr));

		// Set locale to en_US
		std::setlocale(LC_NUMERIC, "en_US");

		XMLReader reader((uint8_t*)bytes.data(), bytes.size());

		while (reader.Inspect())
		{
			if (reader.GetInspected() == XMLReader::Inspected::StartTag)
			{
				if (reader.GetLocalName() == "configs")
					config(reader);
				else
				{
					LogWarning("Unknown config : ", reader.GetLocalName());
					reader.skipCurrentElement();
				}
			}
		}

		// Restore original locale
		std::setlocale(LC_NUMERIC, saved_locale);
	}
}
