// message.h
#ifndef MESSAGE_H
#define MESSAGE_H

#include <map>
#include <string>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

enum class Type
{
    QUERY,
    HELLO,
    FINISHED,
    REGISTER,
    PROFILE_DATA,
    WARMING_DONE,
    STOP,
    DEPLOY,
};

struct MessageTypeInfo
{
    std::string name;
    int value;
};

std::map<Type, std::string> type2string = {
    {Type::QUERY, "QUERY"},
    {Type::HELLO, "HELLO"},
    {Type::FINISHED, "FINISHED"},
    {Type::REGISTER, "REGISTER"},
    {Type::PROFILE_DATA, "PROFILE_DATA"},
    {Type::WARMING_DONE, "WARMING_DONE"},
    {Type::STOP, "STOP"},
    {Type::DEPLOY, "DEPLOY"},
};

class Message
{
public:
    Message() {}

    Message(const std::string &type)
        : type_(type) {}

    Message(const std::string &type, const std::map<std::string, std::string> &data)
        : type_(type), data_(data)
    {
    }

    Message(const float timestamp, const std::string &type, const std::map<std::string, std::string> &data)
        : timestamp_(timestamp), type_(type), data_(data) {}

    float getTimestamp() const { return timestamp_; }
    std::string getType() const { return type_; }
    std::map<std::string, std::string> get_data() const { return data_; }

    void append_data(std::string key, std::string value)
    {
        data_[key] = value;
    }

    std::string serialize() const
    {
        // convert to JSON: copy each value into the JSON object
        json j = {{"timestamp", timestamp_}, {"type", type_}, {"data", data_}};
        return j.dump();
    }
    
    void deserialize(const std::string &s)
    {
        auto j = json::parse(s);
        
        // convert from JSON: copy each value from the JSON object
        data_       = j["data"].get<std::map<std::string, std::string>>();
        type_       = j["type"].get<std::string>();
        timestamp_  = j["timestamp"].get<float>();
    }

    std::string to_string() const
    {
        std::string data_format = "[";
        for (const auto &[key, value]: data_)
            data_format += "'" + key + "': " + value + ", ";
        data_format += "]";
        return "Message('timestamp': " + std::to_string(timestamp_) +
               ", 'type': " + type_ +
               ", 'data': " + data_format + ")";
    }

private:
    float timestamp_ = 0.0;
    std::string type_;
    std::map<std::string, std::string> data_;
};

#endif // MESSAGE_H